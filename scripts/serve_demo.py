# -*- coding: utf-8 -*-
"""
================================================================================
本地打分服务  serve_demo.py
================================================================================

用途
----
把已训练好的搭配兼容性模型（均值池化 + MLP）包装成一个本地 HTTP 服务，供前端联调。

提供的接口：
    GET  /health              健康检查
    POST /api/outfit/score    搭配兼容性打分
    POST /api/outfit/fill     缺件推荐（补全搭配）
    GET  /api/item/{item_id}  查询商品是否有特征
    POST /api/search          商品检索（文字搜图 / 以图搜图）
    GET  /api/quality/{item}  质量评估（评分统计 + 评论维度优缺点）
    POST /api/recommend       按需求推荐（预算/偏好特征，同学 Fitin 包）
    POST /api/chat            自然语言路由（推荐 / 质量 / 搭配 / 搜索）

设计原则
--------
    * 零额外依赖：只用 Python 标准库 http.server + numpy + torch（项目已装）；
    * 只读，不修改任何特征文件或模型文件；
    * 模型输出 logit 经 sigmoid 转为 0~1 分数；
    * 特征按 item_id 索引（绝不按 Parquet 行号），多个划分特征目录合并成一个查找表。

运行方式
--------
  python -X utf8 scripts/serve_demo.py
  python -X utf8 scripts/serve_demo.py --port 8000 --host 0.0.0.0
  python -X utf8 scripts/serve_demo.py --model outputs/baseline_v1_final/best_model.pt

请求示例（另开一个终端用 curl 或 Postman 试）：
  curl -X POST http://127.0.0.1:8000/api/outfit/score -H "Content-Type: application/json" -d "{\"items\":[\"132621870\",\"153967122\",\"171169800\"]}"
================================================================================
"""

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from llm_agent import LLMAgent

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ------------------------------------------------------------------------------
# 路径默认值
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "baseline_v1_final" / "best_model.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "outputs" / "baseline_v1_final" / "config.json"
# 商品库（真实数据）相关默认路径：搜索 / 质量评估接口依赖
DEFAULT_CATALOG_FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features_catalog"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
DEFAULT_QUALITY_CACHE = PROJECT_ROOT / "data" / "processed" / "quality_cache.json"
DEFAULT_PRICE_PROVENANCE = PROJECT_ROOT / "data" / "provenance" / "前端价格显示.jsonl"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
# 同学交接的 Fitin 规则推荐/质量维度包（backend_adapter.py + runtime/fitin.sqlite）
DEFAULT_FITIN_ROOT = PROJECT_ROOT / "vendor" / "fitin"

DEFAULT_FEAT_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "clip_features",          # train
    PROJECT_ROOT / "data" / "processed" / "clip_features_valid",    # valid
    PROJECT_ROOT / "data" / "processed" / "clip_features_test",     # test
    DEFAULT_CATALOG_FEAT_DIR,                                       # 商品库（真实数据，用于打分/补全）
]


# ------------------------------------------------------------------------------
# 模型（与训练脚本保持一致）
# ------------------------------------------------------------------------------
class MeanPoolMLP(nn.Module):
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sigmoid(x):
    return 1.0 / (1.0 + float(np.exp(-x)))


# ------------------------------------------------------------------------------
# 加载
# ------------------------------------------------------------------------------
def load_features(feat_dirs):
    """合并多个特征目录，返回 item_id -> 512 维 float32 向量 的字典。"""
    id_to_vec = {}
    for d in feat_dirs:
        d = Path(d)
        feat_path = d / "features.npy"
        ids_path = d / "item_ids.json"
        if not feat_path.exists() or not ids_path.exists():
            continue
        feat = np.load(feat_path)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        for r, iid in enumerate(ids):
            id_to_vec[str(iid)] = feat[r].astype("float32")
    return id_to_vec


def load_model(model_path, config_path):
    """加载模型权重；架构维度优先从 config.json 读，读不到用默认。"""
    if not _HAS_TORCH:
        raise RuntimeError("未安装 torch（pip install torch）")
    in_dim, hidden = 512, 256
    if Path(config_path).exists():
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            arch = cfg.get("arch") or {}
            in_dim = int(arch.get("in_dim", 512))
            hidden = int(arch.get("hidden", 256))
        except Exception:
            pass
    model = MeanPoolMLP(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, in_dim, hidden


# ------------------------------------------------------------------------------
# 打分逻辑
# ------------------------------------------------------------------------------
def outfit_score(model, id_to_vec, items):
    """对一套搭配打分，返回 (score, missing)。score 为 None 表示没有任何有效商品。"""
    vecs, missing = [], []
    for iid in items:
        v = id_to_vec.get(str(iid))
        if v is None:
            missing.append(iid)
        else:
            vecs.append(v)
    if not vecs:
        return None, missing
    z = np.mean(vecs, axis=0).astype("float32")
    with torch.no_grad():
        logit = float(model(torch.from_numpy(z)[None, :]).item())
    return sigmoid(logit), missing


def outfit_fill(model, id_to_vec, items, candidates):
    """缺件推荐：返回按分数降序的 [(item_id, score), ...]，best 为最高分候选。"""
    base = []
    for iid in items:
        v = id_to_vec.get(str(iid))
        if v is not None:
            base.append(v)
    dim = next(iter(id_to_vec.values())).shape[0] if id_to_vec else 512
    base_sum = np.sum(base, axis=0).astype("float32") if base else np.zeros(dim, dtype="float32")
    n = len(base)
    ranked = []
    for c in candidates:
        v = id_to_vec.get(str(c))
        if v is None:
            continue
        z = (base_sum + v) / (n + 1)
        with torch.no_grad():
            logit = float(model(torch.from_numpy(z)[None, :]).item())
        ranked.append((str(c), sigmoid(logit)))
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ------------------------------------------------------------------------------
# 搜索 / 质量 / 对话 相关的辅助函数
# ------------------------------------------------------------------------------
def _decode_base64_image(s):
    """把 base64 字符串（可带 data:image/...;base64, 前缀）解码为 RGB PIL.Image。"""
    from PIL import Image
    if "," in s:
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def _extract_item_ids(message, known_ids):
    """在消息文本里找出出现的已知 item_id（子串匹配）。"""
    found = []
    for iid in known_ids:
        s = str(iid)
        if s and s in message:
            found.append(s)
    return found


# 命中这些词且不含具体 item_id 时，/api/chat 路由到 Fitin 按需推荐（而非 CLIP 搜图）
RECOMMEND_KEYWORDS = (
    "推荐", "建议", "挑", "买", "预算", "元", "以内", "以下", "适合", "想要", "想买",
    "柔软", "透气", "显瘦", "修身", "保暖", "不透", "起球", "缩水", "褪色", "不扎",
    "好打理", "做工", "通勤", "派对", "冬天", "夏天", "尺码",
)


def _init_search(args):
    """尝试加载商品库搜索索引（特征 + metadata + 质量缓存）。失败返回 None，不影响 outfit 接口。"""
    try:
        from build_search import SearchEngine
    except Exception as e:
        print(f"[!] 无法导入 build_search（搜索/质量接口不可用）: {e}")
        return None
    feat_dir = Path(args.search_dir)
    metadata_path = Path(args.metadata)
    quality_path = Path(args.quality_cache)
    price_prov_path = Path(args.price_prov)
    if not (feat_dir / "features.npy").exists() or not (feat_dir / "item_ids.json").exists():
        print(f"[i] 未找到商品库特征 {feat_dir}，搜索/质量接口暂不可用")
        return None
    if not metadata_path.exists():
        print(f"[i] 未找到 metadata {metadata_path}，搜索/质量接口暂不可用")
        return None
    try:
        eng = SearchEngine(
            feat_dir=str(feat_dir),
            metadata_path=str(metadata_path),
            quality_path=str(quality_path) if quality_path.exists() else None,
            price_provenance_path=str(price_prov_path) if price_prov_path.exists() else None,
            model_name=args.clip_model,
            device="cpu",
        )
        print(f"[i] 搜索索引已就绪：{eng.num_items} 件商品（metadata {eng.metadata_count} 件，质量分 {eng.quality_count} 件，价格标签 {len(eng.price_prov)} 件）")
        return eng
    except Exception as e:
        print(f"[!] 搜索索引加载失败（搜索/质量接口不可用）: {e}")
        return None


def _init_fitin(fitin_root=None):
    """加载同学交接的 FitinService（按需推荐 + 评论维度优缺点）。失败返回 None。

    FitinService 纯标准库（sqlite3），读取 vendor/fitin/runtime/fitin.sqlite。
    本服务用单线程 HTTPServer，实例在主线程创建并只被主线程访问，无需加锁。
    """
    fitin_root = Path(fitin_root) if fitin_root else DEFAULT_FITIN_ROOT
    backend_dir = fitin_root
    runtime_dir = fitin_root / "runtime"
    if not (backend_dir / "backend_adapter.py").exists():
        print(f"[i] 未找到 Fitin 后端包 {backend_dir}，推荐/质量维度接口暂不可用")
        return None
    if not (runtime_dir / "fitin.sqlite").exists():
        print(f"[i] 未找到 {runtime_dir / 'fitin.sqlite'}，推荐/质量维度接口暂不可用")
        return None
    try:
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from backend_adapter import FitinService
        svc = FitinService(data_root=str(runtime_dir))
        print(f"[i] Fitin 推荐/质量服务已就绪（{runtime_dir / 'fitin.sqlite'}）")
        return svc
    except Exception as e:
        print(f"[!] Fitin 服务加载失败（推荐/质量维度接口不可用）: {e}")
        return None


def _fitin_quality(fitin, item_id):
    """从 Fitin 服务取一件商品的维度优缺点（pros/cons + 维度统计）。失败返回 {}。

    用 FitinService.product() 拿公开的优缺点文本，用 catalog.get_product_assessment()
    拿维度正负面统计（只保留有提及的维度），星级统计仍以 quality_cache.json 为准。
    """
    if fitin is None:
        return {}
    try:
        prod = fitin.product(item_id)
        detail = fitin.catalog.get_product_assessment(item_id)
        assessment = detail["assessment"]
        aspects = {}
        for a in assessment.get("aspects", {}).values():
            if a.get("mentioned_reviews"):
                aspects[a["name"]] = {
                    "mentioned": a.get("mentioned_reviews", 0),
                    "positive": a.get("positive_reviews", 0),
                    "negative": a.get("negative_reviews", 0),
                    "mixed": a.get("mixed_reviews", 0),
                    "neutral": a.get("neutral_reviews", 0),
                }
        return {
            "pros": prod.get("pros", []),
            "cons": prod.get("cons", []),
            "aspects": aspects,
            "evidence_level": assessment.get("evidence_level_zh"),
            "assessment_status": assessment.get("assessment_status"),
        }
    except Exception:
        return {}


# ------------------------------------------------------------------------------
# HTTP 服务
# ------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    model = None
    id_to_vec = None
    search = None
    fitin = None
    llm = None
    model_name = "MeanPoolMLP"

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return None

    # ------------------------------------------------------------------ LLM 工具
    def _execute_tool(self, name, args):
        """LLM function calling 的本地执行器：按工具名调用已有逻辑，返回结构化结果。"""
        if name == "search_products":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return {"error": "缺少搜索词 query"}
            if self.search is None:
                return {"error": "搜索服务未启用"}
            top_k = args.get("top_k", 5)
            if not isinstance(top_k, int) or not (1 <= top_k <= 20):
                top_k = 5
            return {"results": self.search.search_text(query.strip(), top_k=top_k)}

        if name == "recommend_products":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return {"error": "缺少需求描述 query"}
            if self.fitin is None:
                return {"error": "推荐服务未启用"}
            limit = args.get("limit", 3)
            if not isinstance(limit, int) or not (1 <= limit <= 20):
                limit = 3
            return self.fitin.recommend(query.strip(), limit=limit)

        if name == "get_quality":
            item_id = str(args.get("item_id", "")).strip()
            if not item_id:
                return {"error": "缺少商品编号 item_id"}
            if self.search is None:
                return {"error": "质量服务未启用"}
            meta = self.search.metadata.get(item_id, {})
            q = self.search.quality.get(item_id, {})
            if not meta and not q:
                return {"error": f"商品 {item_id} 不存在"}
            fit = _fitin_quality(self.fitin, item_id)
            prov = self.search.price_prov.get(item_id, {})
            return {
                "item_id": item_id,
                "title": meta.get("title"),
                "rating_mean_5": q.get("rating_mean_5"),
                "rating_count": q.get("rating_count"),
                "positive_rate": q.get("positive_rate"),
                "pros": fit.get("pros", []),
                "cons": fit.get("cons", []),
                "price_display": prov.get("price_display"),
            }

        if name == "score_outfit":
            items = args.get("items") or []
            if not isinstance(items, list) or len(items) < 2:
                return {"error": "items 至少需要 2 个商品编号"}
            score, missing = outfit_score(self.model, self.id_to_vec, items)
            if score is None:
                return {"error": "所有商品均无特征", "missing": [str(x) for x in missing]}
            return {"score": round(score, 4), "items": [str(x) for x in items],
                    "missing": [str(x) for x in missing]}

        if name == "fill_outfit":
            items = args.get("items") or []
            candidates = args.get("candidates") or []
            if not isinstance(items, list) or not items or not isinstance(candidates, list) or not candidates:
                return {"error": "items 和 candidates 都必须是非空列表"}
            ranked = outfit_fill(self.model, self.id_to_vec, items, candidates)
            if not ranked:
                return {"error": "候选均无特征"}
            return {"best_item_id": ranked[0][0], "best_score": round(ranked[0][1], 4),
                    "ranked": [{"item_id": i, "score": round(s, 4)} for i, s in ranked[:5]]}

        return {"error": f"未知工具 {name}"}

    # ------------------------------------------------------------------ 规则路由（LLM 未配置或失败时的保底）
    def _rule_chat(self, message):
        actions = []
        known_ids = self.search.ids if self.search else []
        ids = _extract_item_ids(message, known_ids)

        if ids and any(k in message for k in ("评价", "质量", "评分", "口碑", "好不好", "怎么样")):
            item_id = ids[0]
            q = self.search.quality.get(item_id, {})
            meta = self.search.metadata.get(item_id, {})
            fit = _fitin_quality(self.fitin, item_id)
            actions.append({
                "type": "quality",
                "item_id": item_id,
                "rating_mean": q.get("rating_mean"),
                "rating_mean_5": q.get("rating_mean_5"),
                "rating_count": q.get("rating_count"),
                "positive_rate": q.get("positive_rate"),
                "aspects": fit.get("aspects", {}),
                "pros": fit.get("pros", []),
                "cons": fit.get("cons", []),
                "title": meta.get("title"),
            })
            return f"已为你查看商品 {item_id} 的质量信息。", actions
        if len(ids) >= 2 and any(k in message for k in ("搭", "搭配", "配", "补")):
            score, missing = outfit_score(self.model, self.id_to_vec, ids)
            if score is None:
                return "这些商品都没有特征，无法打分。", actions
            actions.append({
                "type": "outfit",
                "items": [str(x) for x in ids],
                "score": round(score, 4),
                "missing": [str(x) for x in missing],
            })
            return f"这几件商品的搭配度为 {score:.2f}。", actions
        if (not ids and self.fitin is not None
                and any(k in message for k in RECOMMEND_KEYWORDS)):
            try:
                payload = self.fitin.recommend(message, limit=3)
            except Exception as e:
                return f"推荐服务暂时不可用：{e}", actions
            actions.append({"type": "recommend", "payload": payload})
            return payload.get("opening", "已为你挑选了几款商品。"), actions
        if self.search is not None:
            results = self.search.search_text(message, top_k=5)
            actions.append({"type": "search", "results": results})
            return f"已为你找到与「{message}」相关的 {len(results)} 件商品。", actions
        return "搜索服务未启用（缺少商品库数据），暂时只能做搭配打分。", actions

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._send_json({
                "status": "ok",
                "model": self.model_name,
                "num_items": len(self.id_to_vec),
            })
            return
        if path.startswith("/api/item/"):
            item_id = path.split("/api/item/", 1)[1]
            self._send_json({
                "item_id": item_id,
                "has_feature": item_id in self.id_to_vec,
            })
            return
        if path.startswith("/api/quality/"):
            if self.search is None:
                self._send_json({"error": "质量评估服务未启用（缺少商品库数据）"}, 503)
                return
            item_id = path.split("/api/quality/", 1)[1]
            meta = self.search.metadata.get(item_id, {})
            q = self.search.quality.get(item_id, {})
            prov = self.search.price_prov.get(item_id, {})
            if not meta and not q:
                self._send_json({"error": "item not found"}, 404)
                return
            fit = _fitin_quality(self.fitin, item_id)
            self._send_json({
                "item_id": item_id,
                "rating_mean": q.get("rating_mean"),
                "rating_mean_5": q.get("rating_mean_5"),
                "rating_count": q.get("rating_count"),
                "positive_rate": q.get("positive_rate"),
                "rating_dist": q.get("rating_dist", {}),
                "aspects": fit.get("aspects") or q.get("aspects", {}),
                "pros": fit.get("pros") or q.get("pros", []),
                "cons": fit.get("cons") or q.get("cons", []),
                "evidence_level": fit.get("evidence_level"),
                "assessment_status": fit.get("assessment_status"),
                "title": meta.get("title"),
                "price": meta.get("price"),
                "price_display": prov.get("price_display"),
                "price_label": prov.get("price_label"),
                "price_type": prov.get("price_type"),
                "price_is_estimated": prov.get("price_is_estimated"),
                "price_currency": prov.get("price_currency"),
                "semantic_category": meta.get("semantic_category"),
                "image_url": meta.get("image_path"),
            })
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body = self._read_body()
        if body is None:
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        if path == "/api/outfit/score":
            items = body.get("items")
            if not isinstance(items, list) or not items:
                self._send_json({"error": "items 必须是非空列表"}, 400)
                return
            score, missing = outfit_score(self.model, self.id_to_vec, items)
            if score is None:
                self._send_json({"error": "所有商品均无特征", "missing": missing}, 422)
                return
            self._send_json({
                "score": round(score, 4),
                "items": [str(x) for x in items],
                "item_count": len(items),
                "missing": [str(x) for x in missing],
            })
            return

        if path == "/api/outfit/fill":
            items = body.get("items")
            candidates = body.get("candidates")
            if not isinstance(items, list) or not items:
                self._send_json({"error": "items 必须是非空列表"}, 400)
                return
            if not isinstance(candidates, list) or not candidates:
                self._send_json({"error": "candidates 必须是非空列表"}, 400)
                return
            ranked = outfit_fill(self.model, self.id_to_vec, items, candidates)
            if not ranked:
                self._send_json({"error": "所有候选均无特征"}, 422)
                return
            self._send_json({
                "best_item_id": ranked[0][0],
                "best_score": round(ranked[0][1], 4),
                "ranked": [{"item_id": iid, "score": round(s, 4)} for iid, s in ranked],
            })
            return

        if path == "/api/search":
            if self.search is None:
                self._send_json({"error": "搜索服务未启用（缺少商品库特征或 metadata）"}, 503)
                return
            top_k = body.get("top_k", 20)
            if not isinstance(top_k, int) or top_k <= 0:
                top_k = 20
            query = body.get("query")
            image_b64 = body.get("image")
            if isinstance(query, str) and query.strip():
                results = self.search.search_text(query.strip(), top_k=top_k)
            elif isinstance(image_b64, str) and image_b64.strip():
                try:
                    img = _decode_base64_image(image_b64)
                except Exception as e:
                    self._send_json({"error": f"图片解码失败: {e}"}, 400)
                    return
                results = self.search.search_image(img, top_k=top_k)
            else:
                self._send_json({"error": "需要 query（文字）或 image（base64）之一"}, 400)
                return
            self._send_json({"results": results})
            return

        if path == "/api/recommend":
            if self.fitin is None:
                self._send_json({"error": "推荐服务未启用（缺少 Fitin 数据包）"}, 503)
                return
            query = body.get("query")
            if not isinstance(query, str) or not query.strip():
                self._send_json({"error": "query 必须是非空字符串"}, 400)
                return
            limit = body.get("limit", 3)
            if not isinstance(limit, int) or not (1 <= limit <= 20):
                limit = 3
            try:
                payload = self.fitin.recommend(query.strip(), limit=limit)
            except ValueError as e:
                self._send_json({"error": str(e)}, 400)
                return
            except Exception as e:
                self._send_json({"error": f"推荐失败: {e}"}, 500)
                return
            self._send_json(payload)
            return

        if path == "/api/chat":
            message = body.get("message")
            if not isinstance(message, str) or not message.strip():
                self._send_json({"error": "message 必须是非空字符串"}, 400)
                return
            history = body.get("history")
            if history is not None and not isinstance(history, list):
                history = None

            # 优先走 LLM function calling（已配置 DEEPSEEK_API_KEY 时）；否则/失败则回退规则路由
            if Handler.llm is not None and Handler.llm.is_configured():
                try:
                    reply, actions = Handler.llm.run(
                        message.strip(),
                        lambda name, args: self._execute_tool(name, args),
                        history=history,
                    )
                except Exception as e:
                    print(f"[!] LLM 路由失败，回退规则路由: {e}")
                    reply, actions = self._rule_chat(message)
            else:
                reply, actions = self._rule_chat(message)

            self._send_json({"reply": reply, "actions": actions})
            return

        self._send_json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        # 简化日志，只打印请求行
        print("[%s] %s" % (self.log_date_time_string(), format % args))


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="本地搭配打分服务（零依赖）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--feature-dirs", type=str, nargs="*", default=[str(d) for d in DEFAULT_FEAT_DIRS],
                        help="特征目录列表（默认合并 train/valid/test/商品库 四组）")
    parser.add_argument("--search-dir", type=str, default=str(DEFAULT_CATALOG_FEAT_DIR),
                        help="商品库特征目录（默认 clip_features_catalog）")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="商品元数据 csv（默认 data/processed/metadata.csv）")
    parser.add_argument("--quality-cache", type=str, default=str(DEFAULT_QUALITY_CACHE),
                        help="质量评估缓存 json（默认 data/processed/quality_cache.json）")
    parser.add_argument("--price-prov", type=str, default=str(DEFAULT_PRICE_PROVENANCE),
                        help="价格标签 jsonl（默认 data/provenance/前端价格显示.jsonl）")
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
                        help="CLIP 模型名（用于搜索/文字编码）")
    parser.add_argument("--fitin-root", type=str, default=str(DEFAULT_FITIN_ROOT),
                        help="Fitin 规则推荐包根目录（默认 vendor/fitin）")
    args = parser.parse_args()

    if not _HAS_TORCH:
        print("!! 未安装 torch（pip install torch）")
        raise SystemExit(1)

    print("=" * 60)
    print("本地搭配打分服务")
    print("=" * 60)

    if not Path(args.model).exists():
        print(f"!! 未找到模型文件: {args.model}")
        print("   请先运行训练脚本生成模型，或指定 --model 路径。")
        raise SystemExit(1)

    print(f"[1] 加载特征 ...")
    id_to_vec = load_features(args.feature_dirs)
    print(f"    共加载 {len(id_to_vec)} 个商品的 512 维特征")
    if not id_to_vec:
        print("!! 未加载到任何特征，请检查 --feature-dirs")
        raise SystemExit(1)

    print(f"[2] 加载模型 ...")
    model, in_dim, hidden = load_model(args.model, args.config)
    print(f"    模型 MeanPoolMLP({in_dim}->{hidden}->1) 已加载并置为 eval 模式")

    Handler.model = model
    Handler.id_to_vec = id_to_vec
    Handler.model_name = "MeanPoolMLP"

    Handler.search = _init_search(args)
    Handler.fitin = _init_fitin(args.fitin_root)
    Handler.llm = LLMAgent()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[3] 服务已启动: http://{args.host}:{args.port}")
    print("    接口:")
    print("      GET  /health")
    print("      POST /api/outfit/score")
    print("      POST /api/outfit/fill")
    print("      GET  /api/item/{item_id}")
    if Handler.search is not None:
        print("      POST /api/search")
        print("      GET  /api/quality/{item_id}")
        print("      POST /api/chat")
    if Handler.fitin is not None:
        print("      POST /api/recommend")
    if Handler.llm.is_configured():
        print(f"      /api/chat 走 LLM 路由（model={Handler.llm.model}）")
    else:
        print("      /api/chat 走规则路由（未配置 DEEPSEEK_API_KEY）")
    print("    按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
