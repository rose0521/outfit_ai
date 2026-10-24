# -*- coding: utf-8 -*-
"""
================================================================================
端到端演示 / 验证脚本  verify_demo.py
================================================================================

把「检索 → 质量评估 → 穿搭打分 → 缺件补全」整条 demo 闭环跑一遍，
直接调用与 serve_demo.py 相同的底层函数（不经过 HTTP），用于：

  1) 快速验证各模块数据已就绪、可互相衔接；
  2) 给演示提供一个可复现的脚本（也便于录屏 / PPT 截图）。

运行方式
--------
  python -X utf8 scripts/verify_demo.py
================================================================================
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_search import SearchEngine
from serve_demo import (DEFAULT_CONFIG, DEFAULT_MODEL, load_features, load_model,
                        outfit_fill, outfit_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "clip_features",
    PROJECT_ROOT / "data" / "processed" / "clip_features_valid",
    PROJECT_ROOT / "data" / "processed" / "clip_features_test",
    PROJECT_ROOT / "data" / "processed" / "clip_features_catalog",
]
METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
QUALITY = PROJECT_ROOT / "data" / "processed" / "quality_cache.json"
PRICE_PROV = PROJECT_ROOT / "data" / "provenance" / "前端价格显示.jsonl"


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # 1) 检索索引
    hr("① 商品检索（CLIP 图文检索）")
    eng = SearchEngine(
        feat_dir=str(FEAT_DIRS[-1]),
        metadata_path=str(METADATA),
        quality_path=str(QUALITY),
        price_provenance_path=str(PRICE_PROV),
        device="cpu",
    )
    print(f"商品库索引：{eng.num_items} 件商品 / {eng.metadata_count} 件有元数据 / {eng.quality_count} 件有质量分")

    for q in ["白色通勤衬衫", "黑色连衣裙", "高腰牛仔裤", "米色针织开衫"]:
        print(f"\n检索「{q}」Top3：")
        for r in eng.search_text(q, top_k=3):
            print(f"  {r['score']:.3f}  [{r['semantic_category']:9}] {r['title']}  "
                  f"{r['price_display']}  评分{r['rating_mean']}（{r['rating_count']}条）")

    # 2) 质量评估
    hr("② 质量评估（评分统计）")
    sample = eng.search_text("黑色连衣裙", top_k=1)[0]["item_id"]
    q = eng.quality.get(sample, {})
    meta = eng.metadata.get(sample, {})
    prov = eng.price_prov.get(sample, {})
    print(f"商品 {sample}：{meta['title']}")
    print(f"  平均分 {q['rating_mean_5']}/5（{q['rating_count']} 条），好评率 {q['positive_rate']:.0%}")
    print(f"  星级分布 {q['rating_dist']}")
    print(f"  价格 {prov.get('price_display')}（{prov.get('price_type')}）")

    # 3) 穿搭打分
    hr("③ 穿搭打分（均值池化 + MLP）")
    model, _, _ = load_model(str(DEFAULT_MODEL), str(DEFAULT_CONFIG))
    id_to_vec = load_features([str(d) for d in FEAT_DIRS])
    print(f"特征查找表共 {len(id_to_vec)} 件（含 Polyvore 训练集 + 商品库）")

    top = eng.search_text("白色衬衫", top_k=1)[0]["item_id"]
    bottom = eng.search_text("牛仔裤", top_k=1)[0]["item_id"]
    outer = eng.search_text("黑色外套", top_k=1)[0]["item_id"]
    outfit = [top, bottom, outer]
    print(f"搭配：{outfit}")
    for iid in outfit:
        print(f"  - {eng.metadata[iid]['title']}  [{eng.metadata[iid]['semantic_category']}]")
    score, missing = outfit_score(model, id_to_vec, outfit)
    print(f"搭配兼容度 = {score:.3f}（0~1，越高越搭），缺失特征 {missing or '无'}")

    # 4) 缺件补全
    hr("④ 缺件补全（FITB）")
    base = [top, bottom]          # 已有：上装 + 下装
    candidates = [r["item_id"] for r in eng.search_text("外套", top_k=8)]
    ranked = outfit_fill(model, id_to_vec, base, candidates)
    print(f"已有单品：{base}")
    print("补一件外套的候选排名（按搭配度）：")
    for iid, s in ranked[:5]:
        print(f"  {s:.3f}  {eng.metadata[iid]['title']}")

    hr("验证通过：检索 / 质量 / 打分 / 补全 四环节数据与逻辑均正常")


if __name__ == "__main__":
    main()
