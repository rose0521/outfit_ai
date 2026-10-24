# -*- coding: utf-8 -*-
"""
================================================================================
商品检索模块（CLIP 文字搜图 / 以图搜图）  build_search.py
================================================================================

用途
----
基于商品库的 CLIP 图片特征做检索。文字和图片被 CLIP 映射到同一个 512 维空间，
图片特征已 L2 归一化，因此「查询向量 ⊙ 图片特征」的内积就是余弦相似度。

两种用法
--------
1) 命令行自测（数据就绪后）：
     python -X utf8 scripts/build_search.py --query "白色通勤衬衫" --top-k 5
     python -X utf8 scripts/build_search.py --image data/processed/images/100001.jpg --top-k 5

2) 被 serve_demo.py import（/api/search 接口），见 SearchEngine 类。

依赖
----
torch / transformers / Pillow / numpy（与项目一致）。
================================================================================
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from clip_utils import DEFAULT_MODEL, encode_images_pil, encode_texts, load_clip, load_image
from zh2en import translate_zh2en

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features_catalog"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
DEFAULT_QUALITY = PROJECT_ROOT / "data" / "processed" / "quality_cache.json"
DEFAULT_PRICE_PROVENANCE = PROJECT_ROOT / "data" / "provenance" / "前端价格显示.jsonl"


class SearchEngine:
    """加载商品库特征 + 元数据 + 质量缓存，提供文字 / 图片检索。"""

    def __init__(self, feat_dir, metadata_path, quality_path=None,
                 price_provenance_path=None,
                 model_name=DEFAULT_MODEL, device="cpu"):
        feat_dir = Path(feat_dir)
        self.feats = np.load(feat_dir / "features.npy").astype("float32")
        self.ids = [str(x) for x in json.loads((feat_dir / "item_ids.json").read_text(encoding="utf-8"))]
        if self.feats.shape[0] != len(self.ids):
            raise ValueError("features.npy 与 item_ids.json 行数不一致")

        self.metadata = self._load_metadata(metadata_path)
        self.quality = self._load_quality(quality_path)
        self.price_prov = self._load_price_provenance(price_provenance_path)

        # 只检索有元数据（能展示）的商品；若 metadata 缺失则全部检索
        if self.metadata:
            mask = np.array([i in self.metadata for i in self.ids], dtype=bool)
            self.valid_mask = mask if mask.any() else np.ones(len(self.ids), dtype=bool)
        else:
            self.valid_mask = np.ones(len(self.ids), dtype=bool)

        self.model, self.processor = load_clip(model_name, device)
        self.device = device

    @property
    def num_items(self):
        return len(self.ids)

    @property
    def metadata_count(self):
        return len(self.metadata)

    @property
    def quality_count(self):
        return len(self.quality)

    # ------------------------------------------------------------------ 加载
    @staticmethod
    def _load_metadata(metadata_path):
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            return {}
        out = {}
        with open(metadata_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                iid = str(row.get("item_id", "")).strip()
                if not iid:
                    continue
                item = {k: (v.strip() if isinstance(v, str) else v)
                        for k, v in row.items() if k != "item_id"}
                if "price" in item and item["price"] not in (None, ""):
                    try:
                        item["price"] = float(item["price"])
                    except (TypeError, ValueError):
                        pass
                out[iid] = item
        return out

    @staticmethod
    def _load_quality(quality_path):
        if not quality_path:
            return {}
        quality_path = Path(quality_path)
        if not quality_path.exists():
            return {}
        data = json.loads(quality_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            out = {}
            for r in data:
                iid = str(r.get("item_id", "")).strip()
                if iid:
                    out[iid] = {k: v for k, v in r.items() if k != "item_id"}
            return out
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
        return {}

    @staticmethod
    def _load_price_provenance(price_provenance_path):
        """读 前端价格显示.jsonl，返回 {item_id: {price_display, price_type, ...}}。

        来源文件逐行是 JSON（每件商品一条），字段含 price_display / price_type /
        price_label / price_is_estimated / price_currency / price_note。必须随商品信息
        返回前端，避免把估算价当成真实售价（见 FITIN 交接说明）。
        """
        if not price_provenance_path:
            return {}
        price_provenance_path = Path(price_provenance_path)
        if not price_provenance_path.exists():
            return {}
        out = {}
        with open(price_provenance_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = str(r.get("item_id", "")).strip()
                if iid:
                    out[iid] = {k: v for k, v in r.items() if k != "item_id"}
        return out

    # ------------------------------------------------------------------ 检索
    def _rank(self, q, top_k):
        scores = self.feats @ q          # 内积 = 余弦相似度（均已 L2 归一化）
        scores = scores.copy()
        scores[~self.valid_mask] = -np.inf
        order = np.argsort(-scores)[:top_k]
        return order, scores

    def search_text(self, query, top_k=20):
        # 中文查询先译成英文，否则 CLIP 英文文本编码器对中文几乎无区分度
        q_text = translate_zh2en(query)
        q = encode_texts(self.model, self.processor, [q_text], self.device)[0]
        order, scores = self._rank(q, top_k)
        return self._format(order, scores)

    def search_image(self, image, top_k=20):
        if isinstance(image, (str, Path)):
            image = load_image(image)
        q = encode_images_pil(self.model, self.processor, [image], self.device)[0]
        order, scores = self._rank(q, top_k)
        return self._format(order, scores)

    def _format(self, order, scores):
        results = []
        for idx in order:
            i = int(idx)
            iid = self.ids[i]
            meta = self.metadata.get(iid, {})
            q = self.quality.get(iid, {})
            prov = self.price_prov.get(iid, {})
            results.append({
                "item_id": iid,
                "score": round(float(scores[i]), 4),
                "title": meta.get("title"),
                "price": meta.get("price"),
                "price_display": prov.get("price_display"),
                "price_label": prov.get("price_label"),
                "price_type": prov.get("price_type"),
                "price_is_estimated": prov.get("price_is_estimated"),
                "price_currency": prov.get("price_currency"),
                "image_url": meta.get("image_path"),
                "semantic_category": meta.get("semantic_category"),
                "rating_mean": q.get("rating_mean"),
                "rating_count": q.get("rating_count"),
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="商品检索自测（文字搜图 / 以图搜图）")
    parser.add_argument("--feat-dir", type=str, default=str(DEFAULT_FEAT_DIR))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--quality", type=str, default=str(DEFAULT_QUALITY))
    parser.add_argument("--price-prov", type=str, default=str(DEFAULT_PRICE_PROVENANCE))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.query and not args.image:
        parser.error("必须提供 --query（文字）或 --image（图片路径）之一")

    eng = SearchEngine(
        feat_dir=args.feat_dir,
        metadata_path=args.metadata,
        quality_path=args.quality if Path(args.quality).exists() else None,
        price_provenance_path=args.price_prov if Path(args.price_prov).exists() else None,
        model_name=args.model,
        device=args.device,
    )

    if args.image:
        results = eng.search_image(args.image, top_k=args.top_k)
    else:
        results = eng.search_text(args.query, top_k=args.top_k)

    print("检索结果：")
    for r in results:
        cat = r.get("semantic_category") or "-"
        rating = f"{r['rating_mean']}" if r.get("rating_mean") is not None else "-"
        price = r.get("price_display") or r.get("price")
        print(f"  {r['item_id']}  score={r['score']}  [{cat}]  {r.get('title')}  价格={price}  评分={rating}")


if __name__ == "__main__":
    main()
