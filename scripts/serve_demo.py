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
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

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
DEFAULT_FEAT_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "clip_features",        # train
    PROJECT_ROOT / "data" / "processed" / "clip_features_valid",  # valid
    PROJECT_ROOT / "data" / "processed" / "clip_features_test",   # test
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
# HTTP 服务
# ------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    model = None
    id_to_vec = None
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
                        help="特征目录列表（默认合并 train/valid/test 三个）")
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

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[3] 服务已启动: http://{args.host}:{args.port}")
    print("    接口:")
    print("      GET  /health")
    print("      POST /api/outfit/score")
    print("      POST /api/outfit/fill")
    print("      GET  /api/item/{item_id}")
    print("    按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
