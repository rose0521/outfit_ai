# 部署到 2核2G Ubuntu 服务器

本文档把 `serve_demo.py` 部署到一台 2 核 2G 内存的 Ubuntu 服务器，供课程 demo 使用。
目标：一条 `python scripts/serve_demo.py` 跑起来，提供检索 / 质量 / 推荐 / 穿搭全套 HTTP 接口。

## 0. 内存与体积预算（2G 够不够）

| 组件 | 内存/体积 | 说明 |
| --- | --- | --- |
| CLIP 模型（openai/clip-vit-base-patch32） | ~605 MB | 加载进内存；文本+图像编码共用 |
| torch（CPU 版）运行时 | ~200 MB | 含 Python 进程开销 |
| 系统（headless Ubuntu） | ~200 MB | |
| 商品库特征（仅 catalog） | ~1 MB | 见下方「砍掉 Polyvore 特征」 |
| 同学 Fitin 包 + SQLite | ~20 MB | 纯标准库 |
| 搭配模型 + 元数据 | ~2 MB | |
| **合计** | **约 1.0~1.1 GB** | 2G 内存可放下，留约 900 MB 余量 |

关键：**只部署商品库特征**（`clip_features_catalog`，500 件、1 MB），
不部署 Polyvore 的 train/valid/test 特征（约 313 MB）。demo 的穿搭打分只针对商品库单品，
用不到 Polyvore 特征。启动时用 `--feature-dirs` 只指到 catalog 目录即可。

## 1. 要拷贝到服务器的文件

服务器上建目录，例如 `/opt/outfit_ai`。**不需要** `.venv/`、`.git/`、`data/raw/`、
`data/FITIN_store/`、`outputs/*/prepared_data.npz`、Polyvore 三个特征目录。

必需内容：

```
scripts/                          # 全部 .py
vendor/fitin/                     # backend_adapter.py + runtime/app/fitin_local + runtime/fitin.sqlite
outputs/baseline_v1_final/best_model.pt
outputs/baseline_v1_final/config.json
data/processed/clip_features_catalog/     # features.npy + item_ids.json（1 MB）
data/processed/metadata.csv
data/processed/reviews.csv
data/processed/quality_cache.json
data/provenance/前端价格显示.jsonl
deploy/                           # 本目录
```

用 rsync（本地 Git Bash / WSL 执行，替换 `SERVER` 为你的服务器地址）：

```bash
rsync -av --progress \
  scripts vendor outputs/baseline_v1_final/best_model.pt \
  outputs/baseline_v1_final/config.json deploy \
  data/processed/clip_features_catalog \
  data/processed/metadata.csv data/processed/reviews.csv \
  data/processed/quality_cache.json \
  data/provenance/前端价格显示.jsonl \
  SERVER:/opt/outfit_ai/
```

也可以用 WinSCP / Xftp 之类的图形工具手动拖，注意别漏了 `vendor/fitin/runtime/fitin.sqlite`
（它被 `.gitignore` 排除，git clone 不会带上）。

> 如果代码已推到 GitHub，也可以服务器上 `git clone https://github.com/rose0521/outfit_ai.git`
> 拿代码和模型权重，再单独 scp 上面几个 `data/` 文件 + `vendor/fitin/runtime/fitin.sqlite`。

## 2. 服务器安装依赖

```bash
cd /opt/outfit_ai
sudo apt update && sudo apt install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch 用 CPU 版（约 200 MB，别装默认的 CUDA 版，那会下 2G+）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 其余依赖（国内可加清华镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r deploy/requirements.txt）
pip install -r deploy/requirements.txt
```

## 3. 下载 CLIP 模型（一次性）

服务器在国内访问 huggingface.co 通常很慢或被墙，用镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP ok')"
```

模型缓存到 `~/.cache/huggingface`（约 605 MB）。也可以从本地把整个
`~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` 拷到服务器同路径，
免去在线下载。

## 4. 启动

```bash
cd /opt/outfit_ai
source .venv/bin/activate
PORT=8000 bash deploy/serve.sh
```

或前台直跑（注意 `--host 0.0.0.0` 才能被外部访问，`--feature-dirs` 只加载商品库特征）：

```bash
python -X utf8 scripts/serve_demo.py --host 0.0.0.0 --port 8000 \
  --feature-dirs data/processed/clip_features_catalog
```

验证：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/recommend -H 'Content-Type: application/json' \
  -d '{"query":"预算300元以内买柔软的连衣裙","limit":3}'
```

## 5. 用 systemd 常驻（推荐）

把 `deploy/outfit-ai.service` 里的路径改成你的实际路径后：

```bash
sudo cp deploy/outfit-ai.service /etc/systemd/system/outfit-ai.service
sudo systemctl daemon-reload
sudo systemctl enable --now outfit-ai
sudo systemctl status outfit-ai      # 看日志
journalctl -u outfit-ai -f           # 实时日志
```

## 6. 对外访问

- 直接开 8000 端口（云厂商安全组放行 8000）即可，demo 够用。
- 想用 80/443 + 域名，就在前面加 nginx 反代到 127.0.0.1:8000，这里不展开。
- 商品图片是 Amazon 的 URL，由**浏览器**直接加载，服务器不用访问 Amazon。
- 服务器只在「装依赖」和「首次下 CLIP 模型」时需要外网，之后全程离线可跑。

## 7. 已知限制与降级

- 服务是单线程 `HTTPServer`：一次只处理一个请求。2 核 CPU + CLIP 推理本来就吃单核，
  demo 规模够用；要并发就把 CLIP 推理放进独立进程/加锁换 `ThreadingHTTPServer`（注意
  同学 Fitin 的 SQLite 连接不能跨线程，需加锁 + `check_same_thread=False`）。
- 2G 内存不要开 swap 意外同时起两个服务；重启时 CLIP 模型加载要几秒到几十秒属正常。
- 若内存仍紧张，可把 `clip_utils.py` 的 `load_clip` 改成只加载 text 侧（约省 350 MB），
  代价是「以图搜图」失效；文字搜图和穿搭打分不受影响。
