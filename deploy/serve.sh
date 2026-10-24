#!/usr/bin/env bash
# 启动 outfit_ai 服务（2核2G 服务器精简版：只加载商品库特征，不加载 Polyvore 特征）
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到项目根目录

# 如果存在 .venv 就用它，否则用系统 python（已装好依赖）
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "[serve] 启动: http://${HOST}:${PORT}（特征仅 catalog）"
exec "$PY" -X utf8 scripts/serve_demo.py \
  --host "$HOST" \
  --port "$PORT" \
  --feature-dirs data/processed/clip_features_catalog
