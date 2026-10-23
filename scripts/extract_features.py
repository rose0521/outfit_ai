# -*- coding: utf-8 -*-
"""
================================================================================
模块 B：CLIP 图片特征提取  extract_features.py
================================================================================

用途
----
使用预训练 CLIP（openai/clip-vit-base-patch32）对 Parquet 中内嵌的图片字节
提取图片特征，用于后续的穿搭兼容性判断与缺失单品推荐。

先决条件（两个硬门禁，不满足会明确报错退出，绝不静默继续）：
    1. 模块 A（validate_item_mapping.py）必须已通过。本脚本会读取
       outputs/reports/item_mapping_report.json，要求 overall_passed == true。
    2. 全量提取（--full）之前，必须先跑通 smoke test（--smoke-test），
       本脚本会读取 data/processed/clip_features/smoke_test_report.json。

关键实现点
----------
    * CLIP 参数默认冻结：model.eval() + torch.no_grad()，不训练。
    * 图片来源：优先用 image.bytes 解码（Pillow -> RGB），不依赖 image.path 是有效本地路径。
    * 流式读取 Parquet：用 PyArrow iter_batches 分批次读取，绝不一整次把全部图片二进制读进内存。
    * 断点续跑：已提取的 item_id 会跳过，不重复处理；进度/失败记录落盘。
    * 输出（data/processed/clip_features/）：
        - item_ids.json   已提取单品的 ID 列表（与 features 行一一对应）
        - features.npy    特征矩阵 float32，shape=(N, 512)，已做 L2 归一化
        - failed.json     解码/推理失败的记录
        - progress.json   提取进度元信息
        - smoke_test_report.json  smoke test 结果

依赖安装（在项目所用 Python 环境里执行，先确认 python 指向正确环境）：
    # 仅 CPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    # 或 NVIDIA GPU（CUDA 12.1，按你的显卡/驱动选对应版本）
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # 其余依赖
    pip install transformers Pillow numpy tqdm pyarrow

运行方式
--------
  # 1) 先跑 smoke test（100 张图片，验证模型与数据管线）
  python scripts/extract_features.py --split train --smoke-test

  # 2) smoke test 通过后，明确指定全量提取（支持断点续跑）
  python scripts/extract_features.py --split train --full --batch-size 32

  # 常用参数
  #   --device auto|cpu|cuda    默认 auto（有 CUDA 用 CUDA，否则 CPU）
  #   --batch-size N            默认 32（每个 worker 的推理 batch 大小）
  #   --num-workers N           默认 2（多进程并发；每个进程各加载一份 CLIP，内存不足降到 1）
  #   --out-dir PATH            默认 data/processed/clip_features

注意
----
  * 模型下载失败时会给出清晰报错（含网络/HF 镜像提示），不会悄悄换其它模型。
  * 本脚本只写 data/processed/ 与 outputs/，不修改任何原始数据文件。
================================================================================
"""

import argparse
import importlib.util
import io
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "polyvore"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features"
DEFAULT_MAPPING_REPORT = PROJECT_ROOT / "outputs" / "reports" / "item_mapping_report.json"

# 划分名 -> Parquet 文件名（json/txt 用 valid，parquet 用 validation，已核实）
PARQUET_FILENAME = {
    "train": "train.parquet",
    "valid": "validation.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}


# ------------------------------------------------------------------------------
# 依赖检查（先于任何第三方 import 执行）
# ------------------------------------------------------------------------------
def check_deps():
    """检查必需第三方库，返回缺失的包名列表（pip 包名）。"""
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "PIL": "Pillow",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "pyarrow": "pyarrow",
    }
    missing = []
    for mod, pkg in required.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)
    return missing


# ------------------------------------------------------------------------------
# 门禁检查
# ------------------------------------------------------------------------------
def check_module_a_passed(report_path: Path):
    """读取模块 A 的报告，要求 overall_passed == true。返回 (ok, msg)。"""
    if not report_path.exists():
        return False, (f"未找到模块 A 报告: {report_path}\n"
                       f"请先运行: python scripts/validate_item_mapping.py")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"模块 A 报告解析失败: {e}"
    if data.get("overall_passed") is not True:
        return False, (f"模块 A 报告显示 overall_passed={data.get('overall_passed')}，"
                       f"映射验证未通过，禁止继续。")
    return True, ""


def check_smoke_passed(out_dir: Path):
    """读取 smoke test 报告，要求 passed == true。返回 (ok, msg)。"""
    p = out_dir / "smoke_test_report.json"
    if not p.exists():
        return False, (f"未找到 smoke test 报告: {p}\n"
                       f"请先运行: python scripts/extract_features.py --smoke-test")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"smoke test 报告解析失败: {e}"
    if data.get("passed") is not True:
        return False, f"smoke test 报告显示 passed={data.get('passed')}，未通过，禁止全量提取。"
    return True, ""


# ------------------------------------------------------------------------------
# 路径与设备
# ------------------------------------------------------------------------------
def resolve_parquet_path(root: Path, split: str):
    """把划分名解析为 Parquet 文件路径。返回 (path, err)。"""
    name = PARQUET_FILENAME.get(split)
    if name is None:
        return None, f"未知 split: {split!r}（支持 train/valid/test）"
    p = root / "data" / "disjoint" / name
    if not p.exists():
        return None, f"Parquet 文件不存在: {p}"
    return p, None


def resolve_device(arg: str):
    """解析 --device，auto 时自动检测 CUDA。返回 torch 设备字符串。"""
    import torch
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg in ("cpu", "cuda"):
        if arg == "cuda" and not torch.cuda.is_available():
            print("!! 警告：指定了 cuda 但当前不可用，回退到 cpu。")
            return "cpu"
        return arg
    return "cpu"


# ------------------------------------------------------------------------------
# 模型加载与推理
# ------------------------------------------------------------------------------
def load_model(model_name: str, device: str):
    """
    加载 CLIP 模型与 processor，冻结参数，返回 (model, processor)。
    下载失败时抛异常（由调用方给出清晰提示，不悄悄切换模型）。
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"模型加载失败: {model_name}\n"
            f"  原始错误: {e}\n"
            f"  可能原因：网络无法访问 HuggingFace，或模型名有误。\n"
            f"  若在中国大陆网络环境，可尝试先设置镜像后重试：\n"
            f"    $env:HF_ENDPOINT='https://hf-mirror.com'\n"
            f"  本脚本不会自动切换到其它模型。") from e

    model.eval()          # 冻结/推理模式
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False  # 显式冻结，确保不训练 CLIP
    return model, processor


def decode_image(image_cell) -> "PIL.Image.Image":
    """把 Parquet 中 image struct 的 bytes 解码为 RGB PIL 图。失败抛异常。"""
    from PIL import Image
    b = image_cell.get("bytes")
    if not b:
        raise ValueError("image.bytes 为空")
    return Image.open(io.BytesIO(b)).convert("RGB")


def extract_features_raw(model, processor, images, device):
    """
    对一个图片列表提取 CLIP 图片特征（未归一化）。
    返回 numpy 数组 shape=(N, projection_dim)。

    实现说明：不同版本的 transformers 中 get_image_features() 的返回值不稳定
    （新版可能返回 BaseModelOutputWithPooling 对象而非张量），因此这里显式走
    vision_model -> pooler_output -> visual_projection，跨版本稳定，且保证得到
    经过 visual_projection 投影的 projection_dim 维特征（clip-vit-base-patch32 为 512）。
    """
    import torch
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        # pooler_output 是 [CLS] 位置的视觉表示 (N, hidden_size=768)
        if hasattr(vision_outputs, "pooler_output"):
            image_embeds = vision_outputs.pooler_output
        else:
            image_embeds = vision_outputs[1]  # 兼容返回 tuple 的情况
        # visual_projection: Linear(768, projection_dim=512)
        feats = model.visual_projection(image_embeds)
    return feats.float().cpu().numpy()


def l2_normalize(x):
    """按行做 L2 归一化（防止除零）。"""
    import numpy as np
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


# ------------------------------------------------------------------------------
# 多进程 worker（Windows 下用 spawn，每个 worker 加载一份 CLIP）
# ------------------------------------------------------------------------------
_WORKER_MODEL = None
_WORKER_PROCESSOR = None
_WORKER_DEVICE = None


def _worker_init(model_name, device, threads):
    """进程池 worker 初始化：设置线程数并加载 CLIP（每个进程一份）。"""
    import torch
    torch.set_num_threads(max(1, int(threads)))
    global _WORKER_MODEL, _WORKER_PROCESSOR, _WORKER_DEVICE
    from transformers import CLIPModel, CLIPProcessor
    _WORKER_MODEL = CLIPModel.from_pretrained(model_name)
    _WORKER_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    _WORKER_MODEL.eval().to(device)
    for p in _WORKER_MODEL.parameters():
        p.requires_grad = False
    _WORKER_DEVICE = device


def _process_items(model, processor, device, items):
    """
    处理一批 (item_id, image_bytes)，返回 (ids, feats, errors)。
    feats 为 float32 且已 L2 归一化的 numpy 数组；无有效图片时为 None。
    """
    from PIL import Image
    ids, imgs, errors = [], [], []
    for item_id, b in items:
        if not b:
            errors.append({"item_id": item_id, "error": "image.bytes 为空"})
            continue
        try:
            imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
            ids.append(item_id)
        except Exception as e:
            errors.append({"item_id": item_id, "error": str(e)})
    feats = None
    if imgs:
        arr = extract_features_raw(model, processor, imgs, device)
        feats = l2_normalize(arr).astype("float32")
    return ids, feats, errors


def _worker_process_batch(items):
    """进程池 worker 入口：处理一个子批次。"""
    return _process_items(_WORKER_MODEL, _WORKER_PROCESSOR, _WORKER_DEVICE, items)


# ------------------------------------------------------------------------------
# smoke test
# ------------------------------------------------------------------------------
def run_smoke_test(model, processor, device, parquet_path: Path, out_dir: Path,
                   batch_size: int, n_smoke: int = 100):
    """
    对前 n_smoke 张图片做 smoke test，执行 5 项检查，写 smoke_test_report.json。
    返回 bool（是否通过）。
    """
    import numpy as np
    import pyarrow.parquet as pq
    from tqdm import tqdm

    pf = pq.ParquetFile(str(parquet_path))
    total_rows = pf.metadata.num_rows
    n = min(n_smoke, total_rows)

    print(f"smoke test：从 {parquet_path.name} 取前 {n} 条记录...")
    batch = next(pf.iter_batches(batch_size=n))
    ids = batch.column("item_id").to_pylist()
    images = batch.column("image").to_pylist()

    # 1) 解码
    dec_ids, dec_images, decode_errors = [], [], []
    for it, im in zip(ids, images):
        try:
            dec_images.append(decode_image(im))
            dec_ids.append(it)
        except Exception as e:
            decode_errors.append({"item_id": it, "error": str(e)})
    decode_ok = (len(decode_errors) == 0)

    dim = None
    expected_dim = int(model.config.projection_dim)
    finite_ok = nonzero_ok = one_to_one_ok = False
    if dec_images:
        raw = extract_features_raw(model, processor, dec_images, device)
        dim = int(raw.shape[1])
        finite_ok = bool(np.isfinite(raw).all())
        nonzero_ok = bool((np.linalg.norm(raw, axis=1) > 0).all())
        one_to_one_ok = (raw.shape[0] == len(dec_ids) == len(set(dec_ids)))
    else:
        raw = None

    checks = {
        "requested": n,
        "decoded": len(dec_images),
        "decode_ok": decode_ok,
        "decode_errors": decode_errors[:10],
        "feature_dim": dim,
        "expected_dim": expected_dim,
        "dim_ok": (dim == expected_dim),
        "finite_ok": finite_ok,
        "nonzero_ok": nonzero_ok,
        "one_to_one_ok": one_to_one_ok,
    }
    passed = bool(decode_ok and checks["dim_ok"] and finite_ok and nonzero_ok and one_to_one_ok)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": getattr(model.config, "name_or_path", None) or "openai/clip-vit-base-patch32",
        "device": device,
        "parquet": str(parquet_path),
        "passed": passed,
        "checks": checks,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印检查结果
    print("-" * 60)
    print(f"  1) 图片解码: {len(dec_images)}/{n} 成功, decode_ok={decode_ok}")
    print(f"  2) 特征维度: {dim} (期望 {expected_dim}), dim_ok={checks['dim_ok']}")
    print(f"  3) 无 NaN/Inf: {finite_ok}")
    print(f"  4) 非零向量: {nonzero_ok}")
    print(f"  5) ID 与特征一一对应: {one_to_one_ok}")
    print("-" * 60)
    print(f"smoke test {'通过' if passed else '未通过'}，报告已写入 {out_dir / 'smoke_test_report.json'}")
    return passed


# ------------------------------------------------------------------------------
# 全量提取
# ------------------------------------------------------------------------------
def _save_state(out_dir: Path, all_ids, feat_parts, failed, dim, meta):
    """把当前特征/ID/失败记录/进度落盘（断点续跑依赖）。"""
    import numpy as np
    feats_path = out_dir / "features.npy"
    ids_path = out_dir / "item_ids.json"
    failed_path = out_dir / "failed.json"
    progress_path = out_dir / "progress.json"

    merged = np.concatenate(feat_parts, axis=0) if feat_parts else np.zeros((0, dim), dtype="float32")
    merged = merged.astype("float32")

    # 原子写：先写临时文件再替换。
    # 注意：临时文件名必须以 .npy 结尾，否则 np.save 会自动追加 .npy 导致文件名对不上。
    tmp = feats_path.with_suffix(".tmp.npy")
    np.save(tmp, merged)
    os.replace(tmp, feats_path)

    ids_path.write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
    failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _stream_parquet_batches(parquet_path: Path, chunk_size: int):
    """
    流式读取 Parquet，每次产出 chunk_size 条 (item_id, image_bytes)。
    绝不一整次把全部图片二进制读进内存。
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet_path))
    for batch in pf.iter_batches(batch_size=chunk_size):
        ids = batch.column("item_id").to_pylist()
        images = batch.column("image").to_pylist()
        items = []
        for it, im in zip(ids, images):
            b = im.get("bytes") if isinstance(im, dict) else None
            items.append((it, b))
        yield items


def run_full_extraction(model, processor, device, parquet_path: Path, out_dir: Path,
                        batch_size: int, checkpoint_every: int,
                        num_workers: int = 1, threads: int = 1,
                        model_name: str = None, dim: int = 512):
    """
    全量提取，支持断点续跑与多进程并发。
    - num_workers == 1：单进程，model/processor 由主进程加载后传入；
    - num_workers > 1：多进程，每个 worker 各自加载一份 CLIP（内存占用 ×num_workers）。
    """
    import numpy as np
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    feats_path = out_dir / "features.npy"
    ids_path = out_dir / "item_ids.json"
    failed_path = out_dir / "failed.json"

    # 续跑：加载已有进度
    existing_ids = []
    existing_feats = None
    failed = []
    if ids_path.exists():
        try:
            existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"!! 读取已有 item_ids.json 失败（将从头开始）: {e}")
            existing_ids = []
    if feats_path.exists() and existing_ids:
        try:
            # 注意：不能用 mmap_mode="r" 加载——Windows 下内存映射会一直占用该文件句柄，
            # 导致后面 os.replace 覆盖 features.npy 时报 PermissionError(拒绝访问)。
            existing_feats = np.load(feats_path)
        except Exception as e:
            print(f"!! 读取已有 features.npy 失败（将从头开始）: {e}")
            existing_feats = None
            existing_ids = []
    if failed_path.exists():
        try:
            failed = json.loads(failed_path.read_text(encoding="utf-8"))
        except Exception:
            failed = []
    done = set(existing_ids)

    all_ids = list(existing_ids)
    feat_parts = [] if existing_feats is None or len(existing_ids) == 0 else [existing_feats]

    pf_total = None
    try:
        import pyarrow.parquet as pq
        pf_total = pq.ParquetFile(str(parquet_path)).metadata.num_rows
    except Exception:
        pf_total = None

    print(f"开始提取: {parquet_path.name}（共 {pf_total} 行，已完成 {len(done)}，"
          f"batch_size={batch_size}，num_workers={num_workers}，device={device}）")

    # 建立进程池（仅多进程时）
    pool = None
    if num_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=num_workers,
                        initializer=_worker_init,
                        initargs=(model_name, device, threads))

    chunk_size = batch_size * max(num_workers, 1)
    t0 = time.time()
    processed_run = 0
    failed_run = 0
    since_checkpoint = 0
    total_seen = 0

    try:
        with tqdm(total=pf_total, desc="提取", unit="row") as pbar:
            for items in _stream_parquet_batches(parquet_path, chunk_size):
                total_seen += len(items)
                # 过滤已完成项；分批分发（多进程）或本地处理（单进程）
                todo = [(i, b) for (i, b) in items if i not in done]
                if todo:
                    if pool is not None:
                        subs = [todo[k::num_workers] for k in range(num_workers)]
                        subs = [s for s in subs if s]
                        results = pool.map(_worker_process_batch, subs)
                    else:
                        results = [_process_items(model, processor, device, todo)]
                    for ids, feats, errors in results:
                        for e in errors:
                            failed.append(e)
                            failed_run += 1
                        if ids:
                            all_ids.extend(ids)
                            feat_parts.append(feats)
                            done.update(ids)
                            processed_run += len(ids)
                            since_checkpoint += len(ids)
                pbar.update(len(items))
                # 定期 checkpoint
                if since_checkpoint >= checkpoint_every:
                    meta = _make_meta(parquet_path, device, dim, total_seen,
                                      len(all_ids), len(failed), t0)
                    _save_state(out_dir, all_ids, feat_parts, failed, dim, meta)
                    since_checkpoint = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # 最终保存
    meta = _make_meta(parquet_path, device, dim, total_seen, len(all_ids), len(failed), t0)
    _save_state(out_dir, all_ids, feat_parts, failed, dim, meta)

    elapsed = time.time() - t0
    print(f"\n提取完成。本次新增成功 {processed_run}，新增失败 {failed_run}，"
          f"累计成功 {len(all_ids)}，累计失败 {len(failed)}，"
          f"耗时 {elapsed:.1f}s，特征维度 {dim}。")
    return {"processed_run": processed_run, "failed_run": failed_run,
            "total_done": len(all_ids), "total_failed": len(failed),
            "elapsed_sec": elapsed, "dim": dim}


def _make_meta(parquet_path, device, dim, total_rows, done, failed, t0):
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parquet": str(parquet_path),
        "device": device,
        "dim": dim,
        "total_rows": total_rows,
        "done": done,
        "failed": failed,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CLIP 图片特征提取（模块 B）")
    parser.add_argument("--split", type=str, default="train",
                        help="划分名：train/valid/test（默认 train）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="模型推理 batch 大小（默认 32）")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cpu|cuda（默认 auto）")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP 模型名（默认 openai/clip-vit-base-patch32）")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="特征输出目录（默认 data/processed/clip_features）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                        help="数据集根目录（默认 data/raw/polyvore）")
    parser.add_argument("--mapping-report", type=str, default=str(DEFAULT_MAPPING_REPORT),
                        help="模块 A 报告路径（默认 outputs/reports/item_mapping_report.json）")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="每新增多少条记录 checkpoint 一次（默认 2000）")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="并发提取的进程数（默认 2；每个进程各自加载一份 CLIP，内存不足请调到 1）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="跑 100 张图 smoke test")
    mode.add_argument("--full", action="store_true", help="全量提取（需 smoke test 已通过）")
    args = parser.parse_args()

    # 1) 依赖检查
    missing = check_deps()
    if missing:
        print("!! 缺少以下依赖，请先安装：")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n安装命令（在项目 Python 环境执行）：")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 或 cu121 GPU 版")
        print("   pip install transformers Pillow numpy tqdm pyarrow")
        sys.exit(1)

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    # 2) 门禁 1：模块 A 必须已通过
    ok, msg = check_module_a_passed(Path(args.mapping_report))
    if not ok:
        print("!! 未通过门禁（模块 A）：" + msg)
        sys.exit(1)

    # 3) 解析 Parquet
    parquet_path, err = resolve_parquet_path(root, args.split)
    if err is not None:
        print(f"!! {err}")
        sys.exit(1)

    # 4) 设备与并发配置
    device = resolve_device(args.device)
    cpu_cores = os.cpu_count() or 1
    num_workers = max(1, int(args.num_workers))
    print(f"使用设备: {device}；模型: {args.model}；num_workers={num_workers}")

    if args.smoke_test:
        # smoke test 单进程，线程拉满
        import torch
        torch.set_num_threads(cpu_cores)
        try:
            model, processor = load_model(args.model, device)
        except RuntimeError as e:
            print("!! " + str(e))
            sys.exit(1)
        passed = run_smoke_test(model, processor, device, parquet_path, out_dir, args.batch_size)
        sys.exit(0 if passed else 1)

    # --full：先过 smoke 门禁
    ok, msg = check_smoke_passed(out_dir)
    if not ok:
        print("!! 未通过门禁（smoke test）：" + msg)
        sys.exit(1)

    # 计算特征维度与每个进程的线程数
    if num_workers <= 1:
        import torch
        torch.set_num_threads(cpu_cores)
        try:
            model, processor = load_model(args.model, device)
        except RuntimeError as e:
            print("!! " + str(e))
            sys.exit(1)
        dim = int(model.config.projection_dim)
        threads = cpu_cores
    else:
        # 多进程：主进程不加载模型（省内存），每个 worker 各自加载；
        # 线程数按进程均分，避免多个进程争抢 CPU。
        from transformers import CLIPConfig
        try:
            dim = int(CLIPConfig.from_pretrained(args.model).projection_dim)
        except Exception as e:
            print(f"!! 读取模型配置失败: {e}")
            sys.exit(1)
        threads = max(1, cpu_cores // num_workers)
        model = processor = None

    run_full_extraction(model, processor, device, parquet_path, out_dir,
                        args.batch_size, args.checkpoint_every, num_workers,
                        threads, args.model, dim)


if __name__ == "__main__":
    main()
