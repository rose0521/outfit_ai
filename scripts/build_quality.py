# -*- coding: utf-8 -*-
"""
================================================================================
质量评估（评分统计层，确定性、零成本）  build_quality.py
================================================================================

用途
----
对 `data/processed/reviews.csv` 做聚合，产出每件商品的质量分缓存：

    data/processed/quality_cache.json

内容（每件一件，键为 item_id）：
    rating_mean      平均分，0~1（除以满分 5）
    rating_mean_5    平均分，1~5（前端可直接显示）
    rating_count     评价条数
    positive_rate    好评率（rating >= 4 占比）
    rating_dist      1~5 星分布计数（供前端画分布条）
    aspects/pros/cons 方面级占位（无 LLM 时为空，前端据此决定是否展示）

这是技术路线文档 §3.4 的第一层（必做）。第二层 LLM 方面级是加分项，
离线预计算，本脚本预留了合并接口：若有 `quality_aspects.json`，会自动并入
aspects/pros/cons（见 `--aspects` 参数），不影响评分统计。

运行方式
--------
  python -X utf8 scripts/build_quality.py
  python -X utf8 scripts/build_quality.py --reviews data/processed/reviews.csv \
      --out data/processed/quality_cache.json
================================================================================
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_REVIEWS = PROJECT_ROOT / "data" / "processed" / "reviews.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "quality_cache.json"


def build_quality(reviews_path: Path):
    """读 reviews.csv，返回 {item_id: quality_dict}。"""
    stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "pos": 0, "dist": {str(i): 0 for i in range(1, 6)}})
    with open(reviews_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            iid = str(row.get("item_id", "")).strip()
            if not iid:
                continue
            try:
                rating = float(row.get("rating"))
            except (TypeError, ValueError):
                continue
            rating = int(round(rating))
            if not (1 <= rating <= 5):
                continue
            s = stats[iid]
            s["sum"] += rating
            s["count"] += 1
            if rating >= 4:
                s["pos"] += 1
            s["dist"][str(rating)] += 1

    out = {}
    for iid, s in stats.items():
        n = s["count"]
        mean_5 = s["sum"] / n
        out[iid] = {
            "rating_mean": round(mean_5 / 5.0, 4),   # 0~1，符合接口契约
            "rating_mean_5": round(mean_5, 2),       # 1~5，前端显示用
            "rating_count": n,
            "positive_rate": round(s["pos"] / n, 4),
            "rating_dist": {k: s["dist"][k] for k in sorted(s["dist"], key=int)},
            "aspects": {},
            "pros": [],
            "cons": [],
        }
    return out


def merge_aspects(quality, aspects_path: Path):
    """若存在方面级缓存（LLM 离线预计算产物），并入 aspects/pros/cons。"""
    if not aspects_path.exists():
        return quality
    data = json.loads(aspects_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for r in data:
            iid = str(r.get("item_id", "")).strip()
            if iid and iid in quality:
                quality[iid]["aspects"] = r.get("aspects", {}) or {}
                quality[iid]["pros"] = r.get("pros", []) or []
                quality[iid]["cons"] = r.get("cons", []) or []
    elif isinstance(data, dict):
        for iid, r in data.items():
            if iid in quality:
                quality[str(iid)]["aspects"] = r.get("aspects", {}) or {}
                quality[str(iid)]["pros"] = r.get("pros", []) or []
                quality[str(iid)]["cons"] = r.get("cons", []) or []
    return quality


def main():
    parser = argparse.ArgumentParser(description="生成商品质量评估缓存（评分统计）")
    parser.add_argument("--reviews", type=str, default=str(DEFAULT_REVIEWS))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--aspects", type=str, default="",
                        help="可选：方面级缓存 json（LLM 离线预计算），并入 aspects/pros/cons")
    args = parser.parse_args()

    reviews_path = Path(args.reviews)
    if not reviews_path.exists():
        print(f"!! 未找到 reviews: {reviews_path}")
        raise SystemExit(1)

    quality = build_quality(reviews_path)
    if args.aspects:
        quality = merge_aspects(quality, Path(args.aspects))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    # 简要校验
    n_items = len(quality)
    n_reviews = sum(q["rating_count"] for q in quality.values())
    has_neg = sum(1 for q in quality.values() if q["rating_mean"] < 0.6)
    print(f"质量缓存已生成：{out_path}")
    print(f"  商品数 {n_items}，评价总数 {n_reviews}")
    print(f"  平均分 < 3 分（rating_mean < 0.6）的商品 {has_neg} 件")


if __name__ == "__main__":
    main()
