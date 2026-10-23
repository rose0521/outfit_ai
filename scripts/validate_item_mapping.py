# -*- coding: utf-8 -*-
"""
================================================================================
模块 A：ID 映射验证脚本  validate_item_mapping.py
================================================================================

用途
----
验证 compatibility_*.txt 中的“完整单品标识”（形如 "199244701_1"）能否通过
“set_id + index” 规则，正确关联到：
    1. disjoint/{split}.json  中的真实 item_id；
    2. data/disjoint/*.parquet 中的 item_id（图片数据）。

映射规则（本脚本要验证的假设，不预设为正确）：
    token = "{set_id}_{index}"
    -> 按最后一个下划线拆出 set_id 与 index
    -> 在搭配 JSON 中找 set_id 对应的穿搭、index 对应的单品
    -> 得到真实 item_id
    -> 校验该 item_id 存在于对应划分的 Parquet 的 item_id 列。

严格判定标准：
    * 一条标签行的**每一个** token 都必须能解析出 item_id，且该 item_id 必须在
      对应划分的 Parquet 中，这一行才算通过；
    * 一个划分的**所有**标签行都通过，该划分才算通过；
    * 任一文件缺失、任一记录无法验证，都明确标记为“未通过”，绝不静默跳过。

只读取 Parquet 的 item_id 列，不读取图片二进制。

运行方式
--------
  python scripts/validate_item_mapping.py
  python scripts/validate_item_mapping.py --root "E:\\code\\outfit_ai\\data\\raw\\polyvore"
  python scripts/validate_item_mapping.py --splits train,valid,test

输出
----
  * 控制台：清晰的分块检查结果；
  * 机器可读报告：outputs/reports/item_mapping_report.json
  * 退出码：全部划分通过返回 0；任一划分未通过返回 1（可用于门禁后续步骤）。

依赖
----
  * Python 3.8+
  * pyarrow（读取 Parquet 的 item_id 列）
  安装：pip install pyarrow

注意
----
  本脚本只读数据，不修改任何原始文件；报告写入 outputs/reports/ 目录。
================================================================================
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# 脚本与项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "polyvore"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "item_mapping_report.json"

# 划分名 -> Parquet 文件名。
# 重要：json/txt 用 "valid"，而 Parquet 用 "validation"（HF 命名习惯）。这是已核实的差异。
PARQUET_FILENAME = {
    "train": "train.parquet",
    "valid": "validation.parquet",
    "test": "test.parquet",
}
SPLIT_NAMES = ["train", "valid", "test"]


# ------------------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------------------
def section(title):
    line = "=" * 80
    print()
    print(line)
    print(title)
    print(line)


def check_file(path: Path):
    """检查单个文件是否存在，返回 {path, exists, size_bytes}。"""
    if path.exists():
        try:
            size = path.stat().st_size
        except Exception:
            size = None
        return {"path": str(path), "exists": True, "size_bytes": size}
    return {"path": str(path), "exists": False, "size_bytes": None}


# ------------------------------------------------------------------------------
# 数据读取
# ------------------------------------------------------------------------------
def load_outfits(json_path: Path):
    """
    读取搭配 JSON（顶层为 list），返回 (outfits, error)。
    outfits: list[dict]，每条含 set_id 与 items（[{item_id, index}, ...]）。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None, "文件不存在"
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}"
    if not isinstance(data, list):
        return None, f"顶层类型不是 list，而是 {type(data).__name__}"
    return data, None


def build_outfit_index(outfits):
    """
    由 outfits 建立映射：
        set_id -> {index: item_id}
    同时统计：重复 set_id、重复 (set_id, index) 对、单品总数。
    返回 (index_map, stats)。
    """
    index_map = {}
    stats = {
        "num_outfits": len(outfits),
        "duplicate_set_ids": 0,
        "duplicate_set_index_pairs": 0,
        "num_item_refs": 0,
    }
    seen_set = set()
    for o in outfits:
        sid = o.get("set_id")
        items = o.get("items")
        if sid is None or not isinstance(items, list):
            continue
        if sid in seen_set:
            stats["duplicate_set_ids"] += 1
        seen_set.add(sid)

        inner = index_map.setdefault(sid, {})
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("index")
            item_id = it.get("item_id")
            if idx is None or item_id is None:
                continue
            stats["num_item_refs"] += 1
            if idx in inner:
                stats["duplicate_set_index_pairs"] += 1
            inner[idx] = item_id
    return index_map, stats


def read_parquet_item_ids(pq_path: Path):
    """
    只读取 Parquet 的 item_id 列，返回 (set_of_item_ids, error)。
    不读取 image 二进制。
    """
    if not _HAS_PYARROW:
        return None, "未安装 pyarrow，无法读取 Parquet（pip install pyarrow）"
    try:
        # columns 只取 item_id，避免把图片二进制读进内存
        tbl = pq.read_table(str(pq_path), columns=["item_id"])
        item_ids = set(tbl.column("item_id").to_pylist())
    except FileNotFoundError:
        return None, "文件不存在"
    except Exception as e:
        return None, f"读取 Parquet 失败: {e}"
    return item_ids, None


def parse_compatibility(txt_path: Path):
    """
    读取 compatibility_*.txt，返回 (lines, stats, error)。
    lines: list[(label, [tokens])]
    stats: 行数、正负样本数、每行 token 数、唯一 token、唯一 set_id。
    """
    stats = {
        "num_lines": 0,
        "num_positive": 0,
        "num_negative": 0,
        "tokens_per_line_min": None,
        "tokens_per_line_max": None,
        "num_unique_tokens": 0,
        "num_unique_set_ids": 0,
    }
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None, stats, "文件不存在"
    except Exception as e:
        return None, stats, f"读取失败: {e}"

    lines = []
    unique_tokens = set()
    unique_sets = set()
    tokens_per_line = []
    for ln in raw:
        parts = ln.split()
        if not parts:
            continue
        label = parts[0]
        tokens = parts[1:]
        lines.append((label, tokens))
        if label == "1":
            stats["num_positive"] += 1
        else:
            stats["num_negative"] += 1
        tokens_per_line.append(len(tokens))
        for t in tokens:
            unique_tokens.add(t)
            if "_" in t:
                unique_sets.add(t.rsplit("_", 1)[0])
    stats["num_lines"] = len(lines)
    stats["num_unique_tokens"] = len(unique_tokens)
    stats["num_unique_set_ids"] = len(unique_sets)
    if tokens_per_line:
        stats["tokens_per_line_min"] = min(tokens_per_line)
        stats["tokens_per_line_max"] = max(tokens_per_line)
    return lines, stats, None


# ------------------------------------------------------------------------------
# 映射验证（核心）
# ------------------------------------------------------------------------------
def validate_mapping(lines, index_map, parquet_ids, num_json_outfits):
    """
    对每一条标签行的每一个 token 执行严格验证。
    返回 (mapping_stats, ok_examples, fail_examples)。

    判定：token -> (set_id, index) -> item_id -> 是否在 parquet_ids。
    任一步失败，该 token 记为失败，整行记为失败行。
    """
    total = 0
    ok = 0
    fail = 0
    fail_lines = 0
    reasons = Counter()
    ok_examples = []
    fail_examples = []

    for label, tokens in lines:
        line_ok = True
        for t in tokens:
            total += 1
            # 1) token 必须含下划线
            if "_" not in t:
                fail += 1
                line_ok = False
                reasons["no_underscore"] += 1
                _push_example(fail_examples, t, "token 不含下划线")
                continue
            set_id, idx_str = t.rsplit("_", 1)
            # 2) index 必须是整数
            try:
                idx = int(idx_str)
            except ValueError:
                fail += 1
                line_ok = False
                reasons["non_int_index"] += 1
                _push_example(fail_examples, t, f"index 不是整数: {idx_str!r}")
                continue
            # 3) set_id 必须在 JSON 中存在
            inner = index_map.get(set_id)
            if inner is None:
                fail += 1
                line_ok = False
                reasons["missing_set_id"] += 1
                _push_example(fail_examples, t, f"set_id={set_id!r} 不在搭配 JSON 中")
                continue
            # 4) index 必须在该 set 中存在
            item_id = inner.get(idx)
            if item_id is None:
                fail += 1
                line_ok = False
                reasons["missing_index"] += 1
                _push_example(fail_examples, t, f"set_id={set_id!r} 中不存在 index={idx}")
                continue
            # 5) item_id 必须在 Parquet 中
            if item_id not in parquet_ids:
                fail += 1
                line_ok = False
                reasons["not_in_parquet"] += 1
                _push_example(fail_examples, t, f"解析出的 item_id={item_id!r} 不在 Parquet 中")
                continue
            # 通过
            ok += 1
            if len(ok_examples) < 5:
                ok_examples.append({
                    "token": t, "set_id": set_id, "index": idx, "item_id": item_id,
                })
        if not line_ok:
            fail_lines += 1

    mapping_stats = {
        "total_tokens": total,
        "ok_tokens": ok,
        "fail_tokens": fail,
        "ok_ratio": (ok / total) if total else 0.0,
        "fail_lines": fail_lines,
        "num_lines": len(lines),
        "fail_reasons": dict(reasons),
        "num_json_outfits": num_json_outfits,
    }
    return mapping_stats, ok_examples, fail_examples


def _push_example(lst, token, reason):
    if len(lst) < 20:
        lst.append({"token": token, "reason": reason})


def check_set_consistency(lines, index_map):
    """
    检查 txt 中出现的 set_id 与 JSON 中的 set_id 是否一致（双向覆盖）。
    返回 (txt_only, json_only, txt_set_ids, json_set_ids)。
    """
    txt_set_ids = set()
    for _, tokens in lines:
        for t in tokens:
            if "_" in t:
                txt_set_ids.add(t.rsplit("_", 1)[0])
    json_set_ids = set(index_map.keys())
    txt_only = sorted(txt_set_ids - json_set_ids)
    json_only = sorted(json_set_ids - txt_set_ids)
    return txt_only, json_only, txt_set_ids, json_set_ids


# ------------------------------------------------------------------------------
# 单划分验证
# ------------------------------------------------------------------------------
def validate_split(root: Path, split: str):
    """
    验证单个划分（train/valid/test），返回结果字典。
    """
    json_path = root / "disjoint" / f"{split}.json"
    txt_path = root / "disjoint" / f"compatibility_{split}.txt"
    pq_name = PARQUET_FILENAME.get(split, f"{split}.parquet")
    pq_path = root / "data" / "disjoint" / pq_name

    result = {
        "split": split,
        "files": {
            f"{split}.json": check_file(json_path),
            f"compatibility_{split}.txt": check_file(txt_path),
            pq_name: check_file(pq_path),
        },
    }

    # 1) 文件存在性
    files_ok = all(v["exists"] for v in result["files"].values())
    if not files_ok:
        result["passed"] = False
        result["error"] = "存在文件缺失，无法验证"
        return result

    # 2) 读 JSON
    outfits, err = load_outfits(json_path)
    if err is not None:
        result["passed"] = False
        result["error"] = f"搭配 JSON 读取失败: {err}"
        return result
    index_map, json_stats = build_outfit_index(outfits)
    result["json_stats"] = json_stats

    # 3) 读 Parquet item_id
    parquet_ids, err = read_parquet_item_ids(pq_path)
    if err is not None:
        result["passed"] = False
        result["error"] = f"Parquet 读取失败: {err}"
        return result
    result["parquet_stats"] = {
        "num_item_ids": len(parquet_ids),
        "filename": pq_name,
    }

    # 4) 读 compatibility txt
    lines, txt_stats, err = parse_compatibility(txt_path)
    if err is not None:
        result["passed"] = False
        result["error"] = f"compatibility 读取失败: {err}"
        return result
    result["txt_stats"] = txt_stats

    # 5) set_id 覆盖一致性（txt 与 JSON 是否一致、是否有重复 set_id）
    txt_only, json_only, txt_set_ids, json_set_ids = check_set_consistency(lines, index_map)
    result["set_consistency"] = {
        "txt_num_unique_set_ids": len(txt_set_ids),
        "json_num_set_ids": len(json_set_ids),
        "txt_only_set_ids": txt_only[:50],
        "json_only_set_ids": json_only[:50],
    }

    # 6) 逐条映射验证（核心）
    mapping_stats, ok_examples, fail_examples = validate_mapping(
        lines, index_map, parquet_ids, json_stats["num_outfits"])
    result["mapping_stats"] = mapping_stats
    result["ok_examples"] = ok_examples
    result["failure_examples"] = fail_examples

    # 7) 最终判定：全部 token 通过 + 无失败行 + 覆盖一致
    all_ok = (mapping_stats["fail_tokens"] == 0
              and mapping_stats["fail_lines"] == 0
              and mapping_stats["total_tokens"] > 0
              and not txt_only and not json_only
              and json_stats["duplicate_set_ids"] == 0
              and json_stats["duplicate_set_index_pairs"] == 0)
    result["passed"] = all_ok
    if not all_ok:
        reasons = []
        if mapping_stats["fail_tokens"]:
            reasons.append(f"{mapping_stats['fail_tokens']} 个 token 映射失败")
        if txt_only:
            reasons.append(f"{len(txt_only)} 个 set_id 仅在 txt 中出现")
        if json_only:
            reasons.append(f"{len(json_only)} 个 set_id 仅在 JSON 中出现")
        if json_stats["duplicate_set_ids"]:
            reasons.append("JSON 中存在重复 set_id")
        if json_stats["duplicate_set_index_pairs"]:
            reasons.append("JSON 中存在重复 (set_id, index) 对")
        result["error"] = "; ".join(reasons)
    return result


# ------------------------------------------------------------------------------
# 跨划分单品重叠检查（额外发现，供训练划分决策参考）
# ------------------------------------------------------------------------------
def cross_split_overlap(root: Path):
    """
    读取三个划分 Parquet 的 item_id，检查两两重叠。
    这是对“disjoint 是否真的单品不重叠”的验证，不参与映射判定，仅作警告。
    """
    sets = {}
    errors = []
    for split in SPLIT_NAMES:
        pq_path = root / "data" / "disjoint" / PARQUET_FILENAME[split]
        ids, err = read_parquet_item_ids(pq_path)
        if err is not None:
            errors.append(f"{split}: {err}")
            sets[split] = set()
        else:
            sets[split] = ids
    overlaps = {}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for a, b in pairs:
        if a in sets and b in sets:
            overlaps[f"{a}_intersect_{b}"] = len(sets[a] & sets[b])
    return {"counts": {k: len(v) for k, v in sets.items()},
            "overlaps": overlaps,
            "errors": errors}


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Polyvore ID 映射验证（模块 A）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                        help="数据集根目录（默认 data/raw/polyvore）")
    parser.add_argument("--splits", type=str, default=",".join(SPLIT_NAMES),
                        help="要验证的划分，逗号分隔（默认 train,valid,test）")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT),
                        help="JSON 报告输出路径（默认 outputs/reports/item_mapping_report.json）")
    args = parser.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report_path = Path(args.report)

    section("[0] 依赖与环境")
    print(f"pyarrow : {'已安装' if _HAS_PYARROW else '未安装（无法读 Parquet）'}")
    print(f"数据根目录: {root}")

    section("[1] 文件存在性检查")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "mapping_rule": (
            "compatibility 中的 token 形如 '{set_id}_{index}'，按最后一个下划线拆分得到 set_id 与 index；"
            "再在搭配 JSON 中查找 set_id 对应穿搭、index 对应的 item_id；"
            "最后校验该 item_id 存在于对应划分 Parquet 的 item_id 列。"
            "注意：token 前缀是 set_id，真实 item_id 是另一个数字，不可直接删除 '_index' 后缀得到。"
        ),
        "parquet_naming_note": "json/txt 划分名为 valid，Parquet 文件名为 validation.parquet（已核实）。",
        "splits": {},
        "cross_split_item_overlap": None,
        "overall_passed": None,
    }

    all_passed = True
    for split in splits:
        section(f"[2] 验证划分: {split}")
        res = validate_split(root, split)
        report["splits"][split] = res

        # 控制台摘要
        print(f"  文件检查: " + ", ".join(
            f"{k.split('/')[-1]}={'OK' if v['exists'] else '缺失'}" for k, v in res["files"].items()))
        if res.get("json_stats"):
            js = res["json_stats"]
            print(f"  JSON: 穿搭数={js['num_outfits']} 单品引用数={js['num_item_refs']} "
                  f"重复set_id={js['duplicate_set_ids']} 重复(set,index)={js['duplicate_set_index_pairs']}")
        if res.get("parquet_stats"):
            print(f"  Parquet({res['parquet_stats']['filename']}): item_id 数={res['parquet_stats']['num_item_ids']}")
        if res.get("txt_stats"):
            ts = res["txt_stats"]
            print(f"  TXT: 行数={ts['num_lines']} 正={ts['num_positive']} 负={ts['num_negative']} "
                  f"唯一token={ts['num_unique_tokens']} 唯一set_id={ts['num_unique_set_ids']}")
        if res.get("mapping_stats"):
            ms = res["mapping_stats"]
            print(f"  映射: token={ms['total_tokens']} 成功={ms['ok_tokens']} 失败={ms['fail_tokens']} "
                  f"成功率={ms['ok_ratio']*100:.4f}% 失败行={ms['fail_lines']}/{ms['num_lines']}")
        if res.get("ok_examples"):
            print("  成功样例:")
            for e in res["ok_examples"][:3]:
                print(f"    {e['token']} -> set_id={e['set_id']}, index={e['index']}, item_id={e['item_id']}")
        if res.get("failure_examples"):
            print("  失败样例:")
            for e in res["failure_examples"][:5]:
                print(f"    {e['token']} : {e['reason']}")
        if not res["passed"]:
            print(f"  !! 未通过: {res.get('error', '')}")
        else:
            print("  => 该划分映射验证通过")
        all_passed = all_passed and res["passed"]

    # 跨划分重叠检查
    section("[3] 跨划分单品重叠检查（额外发现，仅供参考）")
    overlap = cross_split_overlap(root)
    report["cross_split_item_overlap"] = overlap
    print(f"  各划分 Parquet item_id 数: {overlap['counts']}")
    print(f"  两两重叠数: {overlap['overlaps']}")
    if overlap["errors"]:
        print(f"  错误: {overlap['errors']}")
    if any(v > 0 for v in overlap["overlaps"].values()):
        print("  !! 注意：划分之间存在单品重叠，'disjoint' 并非严格单品不重叠，训练/评估时需注意。")

    report["overall_passed"] = all_passed

    # 写报告
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {report_path}")
    except Exception as e:
        print(f"!! 写报告失败: {e}")

    section("[4] 最终结论")
    if all_passed:
        print("全部划分映射验证通过：可以进入模块 B（CLIP 特征提取）。")
        sys.exit(0)
    else:
        print("存在未通过的划分：禁止进入特征提取，请先根据失败样例排查映射。")
        sys.exit(1)


if __name__ == "__main__":
    main()
