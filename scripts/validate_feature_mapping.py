# -*- coding: utf-8 -*-
"""
================================================================================
模块 C：特征与映射端到端验证  validate_feature_mapping.py
================================================================================

用途
----
在模块 A（validate_item_mapping.py：compatibility 标签 ↔ JSON ↔ Parquet 映射）
与模块 B（extract_features.py：CLIP 特征提取）**之后**，对「特征文件本身」以及
「JSON ↔ Parquet ↔ CLIP 特征」三者之间的映射做一次完整审计，为下一阶段
（搭配兼容性训练集的构建）提供可信的数据基础。

本脚本验证（模块 A 未覆盖的部分）：
    1. 特征文件完整性：features.npy 的 shape/dtype/维度、NaN/Inf、零向量、
       L2 归一化、item_ids.json 的数量/唯一性/类型/空串；
    2. 特征 ↔ Parquet 的对应关系：集合是否相等、行顺序是否一致；
    3. train.json ↔ Parquet ↔ 特征 的映射：唯一商品匹配率、双向缺失；
    4. compatibility_*.txt 与 fill_in_blank_*.json 中的标签 token 是否
       都能最终解析到「有特征」的真实 item_id（特征层检查，模块 A 只查到 Parquet）；
    5. 随机抽样若干套搭配，输出人工可读的映射明细。

重要约定（与模块 A 一致，不预设为正确，以实际数据为准）：
    * compatibility token 形如 "{set_id}_{index}"，按最后一个下划线拆出
      set_id 与 index，再在搭配 JSON 中查到真实 item_id（不可直接当商品 ID）；
    * 位置编号从 1 开始（由实际数据确认，脚本会校验连续性并报告）。

设计原则
--------
    * 只读数据，绝不修改原始数据与已有特征文件；
    * 读 Parquet 只取 item_id 列，绝不加载图片二进制；
    * 只依赖 numpy + pyarrow（不依赖 torch/PIL）；
    * 严重映射错误返回非零退出码（1）；「特征行顺序 ≠ Parquet 行顺序」这类
      不影响正确性但影响使用方式的发现，只作为警告输出，不判失败。

运行方式
--------
  python -X utf8 scripts/validate_feature_mapping.py
  python -X utf8 scripts/validate_feature_mapping.py --root "E:\\code\\outfit_ai\\data\\raw\\polyvore"
  python -X utf8 scripts/validate_feature_mapping.py --feature-dir "E:\\code\\outfit_ai\\data\\processed\\clip_features"
  python -X utf8 scripts/validate_feature_mapping.py --splits train --sample-size 20

输出
----
  * 控制台：分块中文审计报告；
  * 机器可读：outputs/reports/feature_mapping_report.json
  * 人类可读：outputs/reports/feature_mapping_report.txt
  * 退出码：全部通过返回 0；存在严重错误返回 1。

依赖
----
  * Python 3.8+
  * numpy、pyarrow
  安装：pip install numpy pyarrow
================================================================================
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# 脚本与项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "polyvore"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_MODULE_A_REPORT = DEFAULT_REPORT_DIR / "item_mapping_report.json"

# 划分名 -> Parquet 文件名（json/txt 用 valid，parquet 用 validation，已核实）
PARQUET_FILENAME = {
    "train": "train.parquet",
    "valid": "validation.parquet",
    "test": "test.parquet",
}
SPLIT_NAMES = ["train", "valid", "test"]

EXPECTED_DIM = 512  # clip-vit-base-patch32 的投影维度


# ------------------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------------------
def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_file(path: Path):
    """检查单个文件是否存在，返回 {path, exists, size_bytes}。"""
    if path.exists():
        try:
            size = path.stat().st_size
        except Exception:
            size = None
        return {"path": str(path), "exists": True, "size_bytes": size}
    return {"path": str(path), "exists": False, "size_bytes": None}


def load_json(path: Path):
    """读取 JSON，返回 (data, error)。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "文件不存在"
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}"
    except Exception as e:
        return None, f"读取失败: {e}"


# ------------------------------------------------------------------------------
# [2] 特征文件完整性审计
# ------------------------------------------------------------------------------
def audit_features(feat_dir: Path, errors, warnings):
    """
    审计 features.npy 与 item_ids.json。
    返回 dict；严重问题写入 errors，一般性发现写入 warnings。
    """
    feat_path = feat_dir / "features.npy"
    ids_path = feat_dir / "item_ids.json"
    failed_path = feat_dir / "failed.json"

    result = {
        "features_npy": check_file(feat_path),
        "item_ids_json": check_file(ids_path),
        "failed_json": check_file(failed_path),
    }

    if not feat_path.exists():
        errors.append("特征文件 features.npy 缺失")
        return result
    if not ids_path.exists():
        errors.append("item_ids.json 缺失")
        return result
    if not _HAS_NUMPY:
        errors.append("未安装 numpy，无法审计特征文件（pip install numpy）")
        return result

    try:
        arr = np.load(feat_path)
    except Exception as e:
        errors.append(f"features.npy 加载失败: {e}")
        return result

    item_ids, err = load_json(ids_path)
    if err is not None:
        errors.append(f"item_ids.json 读取失败: {err}")
        return result
    if not isinstance(item_ids, list):
        errors.append(f"item_ids.json 顶层类型不是 list，而是 {type(item_ids).__name__}")
        return result

    failed, _ = load_json(failed_path)
    if failed is None:
        failed = None  # 缺失时按空处理，但保留提示

    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    ndim = arr.ndim

    nan_cnt = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else -1
    inf_cnt = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else -1

    norms = None
    zero_norm = -1
    norm_min = norm_max = norm_mean = None
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
        norms = np.linalg.norm(arr, axis=1)
        zero_norm = int((norms == 0).sum())
        norm_min = float(norms.min())
        norm_max = float(norms.max())
        norm_mean = float(norms.mean())

    n_ids = len(item_ids)
    all_str = all(isinstance(x, str) for x in item_ids)
    non_empty = all(isinstance(x, str) and x != "" for x in item_ids) if all_str else False
    unique_ids = len(set(item_ids))
    dup_ids = n_ids - unique_ids

    # ---- 判定 ----
    if ndim != 2:
        errors.append(f"features.npy 维度应为 2，实际 {ndim}（shape={shape}）")
    if ndim == 2 and shape[1] != EXPECTED_DIM:
        errors.append(f"特征维度应为 {EXPECTED_DIM}，实际 {shape[1]}")
    if ndim == 2 and shape[0] != n_ids:
        errors.append(f"features.npy 行数 {shape[0]} 与 item_ids.json 数量 {n_ids} 不一致")
    if nan_cnt and nan_cnt > 0:
        errors.append(f"features.npy 存在 NaN：{nan_cnt} 个")
    if inf_cnt and inf_cnt > 0:
        errors.append(f"features.npy 存在 Inf：{inf_cnt} 个")
    if zero_norm > 0:
        errors.append(f"features.npy 存在零向量：{zero_norm} 行")
    if not all_str:
        errors.append("item_ids.json 存在非字符串元素")
    elif not non_empty:
        errors.append("item_ids.json 存在空字符串 ID")
    if dup_ids > 0:
        errors.append(f"item_ids.json 存在重复 ID：{dup_ids} 个（唯一 {unique_ids}/{n_ids}）")

    result.update({
        "shape": list(shape),
        "dtype": dtype,
        "ndim": ndim,
        "num_rows": shape[0] if ndim >= 1 else 0,
        "feature_dim": shape[1] if ndim >= 2 else None,
        "num_item_ids": n_ids,
        "all_str": all_str,
        "non_empty": non_empty,
        "unique_item_ids": unique_ids,
        "duplicate_item_ids": dup_ids,
        "nan_count": nan_cnt,
        "inf_count": inf_cnt,
        "zero_norm_rows": zero_norm,
        "l2_norm_min": norm_min,
        "l2_norm_max": norm_max,
        "l2_norm_mean": norm_mean,
        "failed_records": failed,
    })
    return result


# ------------------------------------------------------------------------------
# [3] Parquet item_id 审计
# ------------------------------------------------------------------------------
def audit_parquet(pq_path: Path, errors):
    """只读 Parquet 的 item_id 列，返回 dict。"""
    if not _HAS_PYARROW:
        errors.append("未安装 pyarrow，无法读取 Parquet（pip install pyarrow）")
        return {"num_rows": None, "num_item_ids": None, "unique_item_ids": None,
                "duplicate_item_ids": None, "id_type": None}
    try:
        pf = pq.ParquetFile(str(pq_path))
        total = pf.metadata.num_rows
        tbl = pq.read_table(str(pq_path), columns=["item_id"])
        ids = tbl.column("item_id").to_pylist()
    except Exception as e:
        errors.append(f"Parquet 读取失败: {e}")
        return {"num_rows": None, "num_item_ids": None, "unique_item_ids": None,
                "duplicate_item_ids": None, "id_type": None}
    n = len(ids)
    uniq = len(set(ids))
    dup = n - uniq
    if dup > 0:
        errors.append(f"Parquet item_id 存在重复：{dup} 个（唯一 {uniq}/{n}）")
    return {
        "num_rows": total,
        "num_item_ids": n,
        "unique_item_ids": uniq,
        "duplicate_item_ids": dup,
        "id_type": type(ids[0]).__name__ if ids else None,
        "_ids": ids,
    }


# ------------------------------------------------------------------------------
# [4] 特征 ↔ Parquet 对应
# ------------------------------------------------------------------------------
def check_feature_parquet_correspondence(feat_ids, pq_ids, errors, warnings):
    """
    检查特征 item_ids 与 Parquet item_ids 的集合相等性与行顺序一致性。
    集合差异判严重错误；行顺序不一致仅告警（不影响正确性，但影响使用方式）。
    """
    feat_set = set(feat_ids)
    pq_set = set(pq_ids)
    feat_only = feat_set - pq_set
    pq_only = pq_set - feat_set
    if feat_only or pq_only:
        errors.append(
            f"特征与 Parquet 的 item_id 集合不一致：特征独有 {len(feat_only)}，"
            f"Parquet 独有 {len(pq_only)}")
    order_match = (list(feat_ids) == [str(x) for x in pq_ids])
    if not order_match:
        mism = sum(1 for a, b in zip(feat_ids, pq_ids) if str(a) != str(b))
        warnings.append(
            f"特征行顺序与 Parquet 行顺序不一致（前 {min(len(feat_ids), len(pq_ids))} 行中 "
            f"{mism} 处不同）。这不影响正确性，但下游**必须通过 item_ids.json 建立 "
            f"item_id -> 特征行 的映射**，不可按位置对齐。")
    return {
        "set_equal": bool(not feat_only and not pq_only),
        "feat_only_count": len(feat_only),
        "parquet_only_count": len(pq_only),
        "feat_only_examples": sorted(feat_only)[:10],
        "parquet_only_examples": sorted(pq_only)[:10],
        "row_order_match": order_match,
    }


# ------------------------------------------------------------------------------
# [5] train.json 结构审计
# ------------------------------------------------------------------------------
def audit_train_json(json_path: Path, errors):
    """审计搭配 JSON 结构，返回 dict（含 index_map 供后续映射）。"""
    outfits, err = load_json(json_path)
    if err is not None:
        errors.append(f"搭配 JSON 读取失败: {err}")
        return None
    if not isinstance(outfits, list):
        errors.append(f"搭配 JSON 顶层类型不是 list，而是 {type(outfits).__name__}")
        return None

    num_outfits = len(outfits)
    index_map = {}          # set_id -> {index: item_id}
    num_item_refs = 0
    dup_set_ids = 0
    dup_set_index_pairs = 0
    seen_set = set()
    item_ref_counter = Counter()      # item_id -> 出现次数
    per_outfit_sizes = Counter()      # 每套商品数 -> 套数
    index_values = set()
    non_contiguous = 0
    min_index = max_index = None

    for o in outfits:
        sid = o.get("set_id")
        items = o.get("items")
        if sid is None or not isinstance(items, list):
            continue
        if sid in seen_set:
            dup_set_ids += 1
        seen_set.add(sid)
        per_outfit_sizes[len(items)] += 1

        inner = index_map.setdefault(sid, {})
        idxs = []
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("index")
            iid = it.get("item_id")
            if idx is None or iid is None:
                continue
            num_item_refs += 1
            item_ref_counter[iid] += 1
            index_values.add(idx)
            idxs.append(idx)
            if idx in inner:
                dup_set_index_pairs += 1
            inner[idx] = iid

        if idxs:
            lo, hi = min(idxs), max(idxs)
            min_index = lo if min_index is None else min(min_index, lo)
            max_index = hi if max_index is None else max(max_index, hi)
            # 位置编号应从 1 开始且连续（1..N），若不符合则记录
            if sorted(idxs) != list(range(1, len(idxs) + 1)):
                non_contiguous += 1

    if dup_set_ids > 0:
        errors.append(f"搭配 JSON 存在重复 set_id：{dup_set_ids} 个")
    if dup_set_index_pairs > 0:
        errors.append(f"搭配 JSON 存在重复 (set_id, index) 对：{dup_set_index_pairs} 个")
    if non_contiguous > 0:
        errors.append(f"存在 {non_contiguous} 套搭配的位置编号不连续（未从 1 连续递增）")

    unique_items = len(item_ref_counter)
    repeated_items = sum(1 for v in item_ref_counter.values() if v > 1)
    max_repeat = max(item_ref_counter.values()) if item_ref_counter else 0

    return {
        "num_outfits": num_outfits,
        "num_item_refs": num_item_refs,
        "unique_item_ids": unique_items,
        "duplicate_set_ids": dup_set_ids,
        "duplicate_set_index_pairs": dup_set_index_pairs,
        "items_per_outfit_distribution": dict(sorted(per_outfit_sizes.items())),
        "index_min": min_index,
        "index_max": max_index,
        "index_non_contiguous_outfits": non_contiguous,
        "items_appearing_in_multiple_outfits": repeated_items,
        "max_item_repeat": max_repeat,
        "index_map": index_map,
        "item_ref_counter": item_ref_counter,
    }


# ------------------------------------------------------------------------------
# [6] JSON ↔ Parquet ↔ 特征 映射
# ------------------------------------------------------------------------------
def check_json_mapping(json_stats, parquet_ids, feat_ids, errors):
    """
    JSON 唯一 item_id 在 Parquet / 特征中的匹配率，及双向缺失统计。
    返回 dict。任一 JSON 商品缺失特征/Parquet 均判严重错误。
    """
    json_items = set(json_stats["item_ref_counter"].keys())
    pq_set = set(parquet_ids)
    feat_set = set(feat_ids)

    in_pq = json_items & pq_set
    json_not_in_pq = json_items - pq_set
    pq_not_in_json = pq_set - json_items

    in_feat = json_items & feat_set
    json_not_in_feat = json_items - feat_set
    feat_not_in_json = feat_set - json_items

    if json_not_in_pq:
        errors.append(f"JSON 中有 {len(json_not_in_pq)} 个商品不在 Parquet（缺图片数据）")
    if json_not_in_feat:
        errors.append(f"JSON 中有 {len(json_not_in_feat)} 个商品没有特征")

    return {
        "json_unique_items": len(json_items),
        "json_in_parquet": len(in_pq),
        "json_parquet_match_ratio": (len(in_pq) / len(json_items)) if json_items else 0.0,
        "json_not_in_parquet": sorted(json_not_in_pq)[:20],
        "json_not_in_parquet_count": len(json_not_in_pq),
        "parquet_not_in_json_count": len(pq_not_in_json),
        "json_in_features": len(in_feat),
        "json_feature_match_ratio": (len(in_feat) / len(json_items)) if json_items else 0.0,
        "json_not_in_features": sorted(json_not_in_feat)[:20],
        "json_not_in_features_count": len(json_not_in_feat),
        "features_not_in_json_count": len(feat_not_in_json),
    }


# ------------------------------------------------------------------------------
# [7] 标签 token -> 特征 映射（compatibility 与 fill_in_blank）
# ------------------------------------------------------------------------------
def _resolve_token_to_item(t, index_map):
    """token "{set_id}_{index}" -> item_id；失败返回 None。"""
    if "_" not in t:
        return None
    set_id, idx_str = t.rsplit("_", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    inner = index_map.get(set_id)
    if inner is None:
        return None
    return inner.get(idx)


def check_compatibility_features(txt_path, index_map, feat_set, errors):
    """compatibility_*.txt 中每个 token 解析到 item_id 后是否都有特征。"""
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        errors.append(f"文件不存在: {txt_path}")
        return None
    except Exception as e:
        errors.append(f"读取失败: {e}")
        return None

    total_tokens = 0
    ok_tokens = 0
    fail_tokens = 0
    fail_examples = []
    n_pos = n_neg = 0
    for ln in raw:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "1":
            n_pos += 1
        else:
            n_neg += 1
        for t in parts[1:]:
            total_tokens += 1
            iid = _resolve_token_to_item(t, index_map)
            if iid is None or iid not in feat_set:
                fail_tokens += 1
                if len(fail_examples) < 20:
                    fail_examples.append({"token": t, "reason":
                        "无法解析" if iid is None else f"item_id={iid} 无特征"})
            else:
                ok_tokens += 1

    if fail_tokens > 0:
        errors.append(f"compatibility 标签中有 {fail_tokens} 个 token 无法解析到有特征的商品")
    return {
        "num_lines": n_pos + n_neg,
        "num_positive": n_pos,
        "num_negative": n_neg,
        "total_tokens": total_tokens,
        "ok_tokens": ok_tokens,
        "fail_tokens": fail_tokens,
        "fail_examples": fail_examples,
    }


def check_fitb_features(fitb_path, index_map, feat_set, errors):
    """fill_in_blank_*.json 中 question 与 answers 的 token 是否都有特征。"""
    data, err = load_json(fitb_path)
    if err is not None:
        errors.append(f"fill_in_blank 读取失败: {err}")
        return None
    if not isinstance(data, list):
        errors.append(f"fill_in_blank 顶层类型不是 list，而是 {type(data).__name__}")
        return None

    total_tokens = ok_tokens = fail_tokens = 0
    fail_examples = []
    for q in data:
        for t in list(q.get("question", [])) + list(q.get("answers", [])):
            total_tokens += 1
            iid = _resolve_token_to_item(t, index_map)
            if iid is None or iid not in feat_set:
                fail_tokens += 1
                if len(fail_examples) < 20:
                    fail_examples.append({"token": t})
            else:
                ok_tokens += 1

    if fail_tokens > 0:
        errors.append(f"fill_in_blank 标签中有 {fail_tokens} 个 token 无法解析到有特征的商品")
    return {
        "num_questions": len(data),
        "total_tokens": total_tokens,
        "ok_tokens": ok_tokens,
        "fail_tokens": fail_tokens,
        "fail_examples": fail_examples,
    }


# ------------------------------------------------------------------------------
# [8] 随机抽样人工可读检查
# ------------------------------------------------------------------------------
def random_sample_check(json_path, feat_set, sample_size, seed=42):
    """随机抽 sample_size 套搭配，输出 set_id/位置/商品ID/特征是否存在的明细。"""
    outfits, err = load_json(json_path)
    if err is not None or not isinstance(outfits, list) or not outfits:
        return [], None
    rng = random.Random(seed)
    sample = rng.sample(outfits, min(sample_size, len(outfits)))
    rows = []
    for o in sample:
        sid = o.get("set_id")
        for it in o.get("items", []):
            rows.append({
                "set_id": sid,
                "index": it.get("index"),
                "item_id": it.get("item_id"),
                "has_feature": (it.get("item_id") in feat_set),
            })
    return rows, None


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="特征与映射端到端验证（模块 C）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                        help="数据集根目录（默认 data/raw/polyvore）")
    parser.add_argument("--feature-dir", type=str, default=str(DEFAULT_FEATURE_DIR),
                        help="特征目录（默认 data/processed/clip_features）")
    parser.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORT_DIR),
                        help="报告输出目录（默认 outputs/reports）")
    parser.add_argument("--module-a-report", type=str, default=str(DEFAULT_MODULE_A_REPORT),
                        help="模块 A 报告路径（默认 outputs/reports/item_mapping_report.json）")
    parser.add_argument("--splits", type=str, default="train",
                        help="要验证的划分，逗号分隔（默认 train；当前特征只提取了 train）")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="随机抽样人工可读检查的套数（默认 20）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    root = Path(args.root)
    feat_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    module_a_report = Path(args.module_a_report)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    section("[0] 依赖与环境")
    print(f"numpy : {'已安装' if _HAS_NUMPY else '未安装'}")
    print(f"pyarrow : {'已安装' if _HAS_PYARROW else '未安装'}")
    print(f"数据根目录: {root}")
    print(f"特征目录: {feat_dir}")
    print(f"报告目录: {report_dir}")
    print(f"验证划分: {splits}")

    # 前置门禁：模块 A 必须已通过（提取特征时已依赖该报告）
    section("[1] 前置门禁：模块 A 映射报告")
    module_a_passed = None
    if module_a_report.exists():
        try:
            ma = json.loads(module_a_report.read_text(encoding="utf-8"))
            module_a_passed = (ma.get("overall_passed") is True)
            print(f"模块 A 报告存在，overall_passed={module_a_passed}")
        except Exception as e:
            print(f"!! 模块 A 报告解析失败: {e}")
            module_a_passed = False
    else:
        print(f"!! 未找到模块 A 报告: {module_a_report}")
        module_a_passed = False

    errors = []       # 严重错误（导致退出码非零）
    warnings = []     # 一般性发现（不判失败）

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "feature_dir": str(feat_dir),
        "module_a_passed": module_a_passed,
        "expected_dim": EXPECTED_DIM,
        "note": (
            "特征行顺序与 Parquet 行顺序可能不一致（多进程提取交错所致）；"
            "下游务必通过 item_ids.json 建立 item_id -> 特征行 映射，不可按位置对齐。"
        ),
        "splits": {},
    }

    if module_a_passed is not True:
        errors.append("模块 A 映射报告缺失或未通过，禁止继续下游训练（请先运行 validate_item_mapping.py）")

    # ---- 特征文件审计（与划分无关，只审计一次）----
    section("[2] 特征文件完整性审计")
    feat_audit = audit_features(feat_dir, errors, warnings)
    report["feature_audit"] = feat_audit
    if feat_audit.get("shape"):
        print(f"  features.npy: shape={feat_audit['shape']} dtype={feat_audit['dtype']}")
        print(f"  NaN={feat_audit['nan_count']}  Inf={feat_audit['inf_count']}  "
              f"零向量={feat_audit['zero_norm_rows']}")
        print(f"  L2 范数: min={feat_audit['l2_norm_min']} max={feat_audit['l2_norm_max']} "
              f"mean={feat_audit['l2_norm_mean']}")
    if feat_audit.get("num_item_ids") is not None:
        print(f"  item_ids.json: 数量={feat_audit['num_item_ids']} "
              f"唯一={feat_audit['unique_item_ids']} 重复={feat_audit['duplicate_item_ids']} "
              f"全str={feat_audit['all_str']} 非空={feat_audit['non_empty']}")
    failed_records = feat_audit.get("failed_records")
    print(f"  failed.json 内容: {failed_records}")

    # 读取特征 item_ids（供后续所有映射使用）
    feat_ids = []
    ids_path = feat_dir / "item_ids.json"
    if ids_path.exists():
        data, _ = load_json(ids_path)
        if isinstance(data, list):
            feat_ids = data

    # ---- 逐划分验证 ----
    for split in splits:
        section(f"[3] 验证划分: {split}")
        json_path = root / "disjoint" / f"{split}.json"
        compat_path = root / "disjoint" / f"compatibility_{split}.txt"
        fitb_path = root / "disjoint" / f"fill_in_blank_{split}.json"
        pq_name = PARQUET_FILENAME.get(split, f"{split}.parquet")
        pq_path = root / "data" / "disjoint" / pq_name

        res = {
            "files": {
                f"{split}.json": check_file(json_path),
                f"compatibility_{split}.txt": check_file(compat_path),
                pq_name: check_file(pq_path),
            },
        }

        files_ok = all(v["exists"] for v in res["files"].values())
        if not files_ok:
            missing = [k for k, v in res["files"].items() if not v["exists"]]
            errors.append(f"[{split}] 存在文件缺失: {missing}")
            report["splits"][split] = res
            continue

        # Parquet 审计
        pq_audit = audit_parquet(pq_path, errors)
        res["parquet_audit"] = {k: v for k, v in pq_audit.items() if k != "_ids"}
        pq_ids = pq_audit.get("_ids", [])
        print(f"  Parquet({pq_name}): 行数={pq_audit['num_rows']} "
              f"唯一={pq_audit['unique_item_ids']} 重复={pq_audit['duplicate_item_ids']} "
              f"类型={pq_audit['id_type']}")

        # 特征 ↔ Parquet 对应
        corr = check_feature_parquet_correspondence(feat_ids, pq_ids, errors, warnings)
        res["feature_parquet_correspondence"] = corr
        print(f"  特征↔Parquet 集合相等: {corr['set_equal']} "
              f"(特征独有={corr['feat_only_count']}, Parquet独有={corr['parquet_only_count']}) "
              f"行顺序一致={corr['row_order_match']}")

        # JSON 结构审计
        json_audit = audit_train_json(json_path, errors)
        if json_audit is None:
            report["splits"][split] = res
            continue
        res["json_audit"] = {k: v for k, v in json_audit.items()
                             if k not in ("index_map", "item_ref_counter")}
        print(f"  JSON: 搭配数={json_audit['num_outfits']} "
              f"单品引用={json_audit['num_item_refs']} "
              f"唯一商品={json_audit['unique_item_ids']}")
        print(f"        每套商品数分布={json_audit['items_per_outfit_distribution']}")
        print(f"        index 范围=[{json_audit['index_min']}, {json_audit['index_max']}] "
              f"不连续套数={json_audit['index_non_contiguous_outfits']}")
        print(f"        跨搭配重复出现的商品={json_audit['items_appearing_in_multiple_outfits']} "
              f"(最大重复 {json_audit['max_item_repeat']} 次)")

        # JSON ↔ Parquet ↔ 特征 映射
        mapping = check_json_mapping(json_audit, pq_ids, feat_ids, errors)
        res["json_mapping"] = mapping
        print(f"  JSON→Parquet 匹配率: {mapping['json_parquet_match_ratio']*100:.4f}% "
              f"({mapping['json_in_parquet']}/{mapping['json_unique_items']})")
        print(f"  JSON→特征   匹配率: {mapping['json_feature_match_ratio']*100:.4f}% "
              f"({mapping['json_in_features']}/{mapping['json_unique_items']})")
        print(f"  JSON 缺 Parquet: {mapping['json_not_in_parquet_count']} | "
              f"Parquet 未被 JSON 使用: {mapping['parquet_not_in_json_count']}")
        print(f"  JSON 缺特征: {mapping['json_not_in_features_count']} | "
              f"特征未被 JSON 使用: {mapping['features_not_in_json_count']}")

        # compatibility 标签 -> 特征
        if compat_path.exists():
            cstats = check_compatibility_features(compat_path, json_audit["index_map"], set(feat_ids), errors)
            res["compatibility_feature_check"] = cstats
            if cstats:
                print(f"  compatibility: 行={cstats['num_lines']} (正={cstats['num_positive']} "
                      f"负={cstats['num_negative']}) token={cstats['total_tokens']} "
                      f"可解析有特征={cstats['ok_tokens']} 失败={cstats['fail_tokens']}")
        else:
            print(f"  compatibility_{split}.txt 不存在（跳过标签特征检查）")

        # fill_in_blank 标签 -> 特征
        if fitb_path.exists():
            fstats = check_fitb_features(fitb_path, json_audit["index_map"], set(feat_ids), errors)
            res["fill_in_blank_feature_check"] = fstats
            if fstats:
                print(f"  fill_in_blank: 题目={fstats['num_questions']} token={fstats['total_tokens']} "
                      f"可解析有特征={fstats['ok_tokens']} 失败={fstats['fail_tokens']}")
        else:
            print(f"  fill_in_blank_{split}.json 不存在（跳过）")

        # 随机抽样人工可读检查
        rows, _ = random_sample_check(json_path, set(feat_ids), args.sample_size, args.seed)
        res["random_sample"] = rows
        if rows:
            missing = [r for r in rows if not r["has_feature"]]
            print(f"  随机抽样 {args.sample_size} 套：共 {len(rows)} 个商品，"
                  f"缺特征 {len(missing)} 个")
        report["splits"][split] = res

    # ---- 汇总 ----
    section("[4] 汇总与判定")
    if errors:
        print(f"发现 {len(errors)} 个严重问题：")
        for e in errors:
            print(f"  !! {e}")
    else:
        print("未发现严重映射错误。")
    if warnings:
        print(f"发现 {len(warnings)} 个提示（不影响正确性）：")
        for w in warnings:
            print(f"  * {w}")

    passed = (len(errors) == 0)
    report["errors"] = errors
    report["warnings"] = warnings
    report["overall_passed"] = passed

    # ---- 写报告 ----
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path_out = report_dir / "feature_mapping_report.json"
    txt_path_out = report_dir / "feature_mapping_report.txt"
    try:
        json_path_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path_out.write_text(_render_txt(report), encoding="utf-8")
        print(f"\n报告已写入:\n  {json_path_out}\n  {txt_path_out}")
    except Exception as e:
        print(f"!! 写报告失败: {e}")

    section("[5] 最终结论")
    if passed:
        print("特征文件与三方映射验证通过，数据可用于构建搭配兼容性训练集。")
        sys.exit(0)
    else:
        print("存在严重映射错误：禁止进入训练，请先按上述失败项排查。")
        sys.exit(1)


def _render_txt(report):
    """把报告渲染为人类可读的中文文本。"""
    lines = []
    lines.append("特征与映射端到端验证报告")
    lines.append("生成时间: " + str(report.get("generated_at", "")))
    lines.append("总体结论: " + ("通过" if report.get("overall_passed") else "未通过"))
    lines.append("模块 A 前置: " + ("通过" if report.get("module_a_passed") else "未通过"))
    lines.append("")
    fa = report.get("feature_audit") or {}
    lines.append("[特征文件]")
    lines.append(f"  shape={fa.get('shape')} dtype={fa.get('dtype')}")
    lines.append(f"  NaN={fa.get('nan_count')} Inf={fa.get('inf_count')} 零向量={fa.get('zero_norm_rows')}")
    lines.append(f"  L2 范数 min={fa.get('l2_norm_min')} max={fa.get('l2_norm_max')} mean={fa.get('l2_norm_mean')}")
    lines.append(f"  item_ids 数量={fa.get('num_item_ids')} 唯一={fa.get('unique_item_ids')} "
                 f"重复={fa.get('duplicate_item_ids')}")
    lines.append("")
    for split, res in (report.get("splits") or {}).items():
        lines.append(f"[划分 {split}]")
        ja = res.get("json_audit") or {}
        lines.append(f"  搭配数={ja.get('num_outfits')} 单品引用={ja.get('num_item_refs')} "
                     f"唯一商品={ja.get('unique_item_ids')}")
        lines.append(f"  每套商品数分布={ja.get('items_per_outfit_distribution')}")
        lines.append(f"  跨搭配重复商品={ja.get('items_appearing_in_multiple_outfits')} "
                     f"(最大重复 {ja.get('max_item_repeat')} 次)")
        m = res.get("json_mapping") or {}
        lines.append(f"  JSON→Parquet 匹配率={m.get('json_parquet_match_ratio')}")
        lines.append(f"  JSON→特征   匹配率={m.get('json_feature_match_ratio')}")
        c = res.get("feature_parquet_correspondence") or {}
        lines.append(f"  特征↔Parquet 集合相等={c.get('set_equal')} 行顺序一致={c.get('row_order_match')}")
        cc = res.get("compatibility_feature_check") or {}
        if cc:
            lines.append(f"  compatibility token 可解析有特征={cc.get('ok_tokens')}/{cc.get('total_tokens')}")
        ff = res.get("fill_in_blank_feature_check") or {}
        if ff:
            lines.append(f"  fill_in_blank token 可解析有特征={ff.get('ok_tokens')}/{ff.get('total_tokens')}")
        lines.append("")
    if report.get("errors"):
        lines.append("[严重问题]")
        lines += [f"  !! {e}" for e in report["errors"]]
    if report.get("warnings"):
        lines.append("[提示]")
        lines += [f"  * {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
