# -*- coding: utf-8 -*-
"""
================================================================================
从 metadata.csv 提取 CLIP 图片特征（图片文件 / URL 版）  extract_features_from_files.py
================================================================================

用途
----
把商品库（真实数据）的图片提取成 CLIP 512 维特征。与 extract_features.py 的区别：
  - extract_features.py 读 Polyvore 的 parquet（内嵌 image.bytes）；
  - 本脚本读 data/processed/metadata.csv 里的图片文件路径或 http(s) URL。

输出（与现有特征目录格式完全一致，下游可统一处理）：
  data/processed/clip_features_catalog/
      item_ids.json   已提取的 item_id（与 features.npy 行一一对应）
      features.npy    float32 (N, 512)，已 L2 归一化
      failed.json     解码 / 下载失败的记录
      progress.json   进度元信息

运行方式
--------
  python -X utf8 scripts/extract_features_from_files.py
  python -X utf8 scripts/extract_features_from_files.py --limit 500
  python -X utf8 scripts/extract_features_from_files.py --device cuda --batch-size 32

说明
----
  * 推理路径与 extract_features.py 完全一致（vision_model -> pooler_output
    -> visual_projection -> L2 归一化），共用 clip_utils.py，特征空间一致；
  * image_path 相对路径按 metadata.csv 所在目录解析；
  * 断点续跑：已存在的 item_id 会跳过；
  * demo 阶段建议 --limit 先只提几百件，CPU 也很快。
================================================================================
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from clip_utils import DEFAULT_MODEL, encode_images_pil, load_clip, load_image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features_catalog"


def check_deps():
    """检查必需第三方库，返回缺失的 pip 包名列表。"""
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "PIL": "Pillow",
        "numpy": "numpy",
    }
    return [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]


def read_metadata(metadata_path: Path, base_dir: Path):
    """读 metadata.csv，返回 [(item_id, image_path), ...]，跳过无图记录。"""
    rows = []
    with open(metadata_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            iid = str(r.get("item_id", "")).strip()
            ip = (r.get("image_path") or r.get("image_url") or "").strip()
            if not iid or not ip:
                continue
            is_url = ip.lower().startswith(("http://", "https://"))
            if not is_url and not Path(ip).is_absolute():
                ip = str(base_dir / ip)
            rows.append((iid, ip))
    return rows


def save_state(out_dir: Path, ids, feats, failed, meta):
    """落盘特征 / ID / 失败记录 / 进度（features.npy 原子写）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "features.tmp.npy"  # 必须以 .npy 结尾，否则 np.save 会追加后缀
    np.save(tmp, feats.astype("float32"))
    os.replace(tmp, out_dir / "features.npy")
    (out_dir / "item_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (out_dir / "failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "progress.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="从 metadata.csv 提取 CLIP 图片特征")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="元数据 csv 路径（列含 item_id, image_path）")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="特征输出目录")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="CLIP 模型名")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cpu|cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每处理多少个 batch 落盘一次（断点续跑用）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条（demo 子集用；0 表示全部）")
    args = parser.parse_args()

    missing = check_deps()
    if missing:
        print("!! 缺少以下依赖，请先安装：" + ", ".join(missing))
        raise SystemExit(1)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"!! 未找到 metadata: {metadata_path}")
        raise SystemExit(1)

    base_dir = metadata_path.resolve().parent
    rows = read_metadata(metadata_path, base_dir)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    print(f"共读取 {len(rows)} 条待提取记录（metadata: {metadata_path}）")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 断点续跑：加载已有进度
    ids_path = out_dir / "item_ids.json"
    existing_ids = []
    existing_feats = None
    if ids_path.exists():
        existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if (out_dir / "features.npy").exists():
            existing_feats = np.load(out_dir / "features.npy")
    done = set(existing_ids)

    failed = []
    failed_path = out_dir / "failed.json"
    if failed_path.exists():
        try:
            failed = json.loads(failed_path.read_text(encoding="utf-8"))
        except Exception:
            failed = []

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}；模型: {args.model}；batch_size={args.batch_size}")

    model, processor = load_clip(args.model, device)
    dim = int(model.config.projection_dim)

    todo = [(iid, ip) for iid, ip in rows if iid not in done]
    print(f"待处理 {len(todo)} 条（已跳过 {len(rows) - len(todo)} 条已完成）")

    all_ids = list(existing_ids)
    feat_parts = [] if existing_feats is None else [existing_feats]

    t0 = time.time()
    processed = 0
    failed_run = 0
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size if todo else 0

    for b in range(n_batches):
        chunk = todo[b * args.batch_size:(b + 1) * args.batch_size]
        imgs, ids_ok = [], []
        for iid, ip in chunk:
            try:
                imgs.append(load_image(ip))
                ids_ok.append(iid)
            except Exception as e:
                failed.append({"item_id": iid, "image": ip, "error": str(e)})
                failed_run += 1
        if imgs:
            feats = encode_images_pil(model, processor, imgs, device)  # 已 L2 归一化
            all_ids.extend(ids_ok)
            feat_parts.append(feats)
            processed += len(ids_ok)
        print(f"  [{b + 1}/{n_batches}] 本批成功 {len(ids_ok)}，累计成功 {processed}，累计失败 {failed_run}")

        if (b + 1) % args.checkpoint_every == 0:
            merged = np.concatenate(feat_parts, axis=0) if feat_parts else np.zeros((0, dim), dtype="float32")
            meta = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "metadata": str(metadata_path), "device": device, "dim": dim,
                    "total": len(rows), "done": len(all_ids), "failed": len(failed)}
            save_state(out_dir, all_ids, merged, failed, meta)

    merged = np.concatenate(feat_parts, axis=0) if feat_parts else np.zeros((0, dim), dtype="float32")
    meta = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadata": str(metadata_path), "device": device, "dim": dim,
            "total": len(rows), "done": len(all_ids), "failed": len(failed)}
    save_state(out_dir, all_ids, merged, failed, meta)

    print(f"\n完成：累计成功 {len(all_ids)}，失败 {len(failed)}，耗时 {time.time() - t0:.1f}s")
    print(f"输出：{out_dir / 'features.npy'}（shape=({len(all_ids)}, {dim})）")


if __name__ == "__main__":
    main()
