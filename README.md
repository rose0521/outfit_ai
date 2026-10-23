# outfit_ai

女装搭配兼容性判断 + 缺件推荐（Fill-In-The-Blank, FITB）。

基于公开数据集 Polyvore 与 CLIP 图片特征，判断一套搭配是否协调（Compatibility, CP），并在"缺了一件"时从候选商品中推荐最搭的一件。

## 模型与效果

- 特征：CLIP（`openai/clip-vit-base-patch32`）512 维，L2 归一化
- 模型：均值池化 + MLP（512 → 256 → 1），约 13 万参数，CPU 可运行
- 官方测试集指标（seed 42）：

| 指标 | 数值 |
| --- | --- |
| 搭配兼容性 CP-AUC | 0.8621 |
| 缺件推荐 FITB Top-1 | 0.5743（随机四选一为 0.25） |

> 这是第一版基线，指标仍有提升空间。

## 目录结构

```
outfit_ai/
├── scripts/     # 特征提取、校验、训练、本地打分服务
├── docs/        # 前端对接说明、训练计划
├── outputs/     # 训练产物（模型权重、评估报告；npz 缓存已忽略）
└── data/        # 原始数据与特征（已忽略，需本地准备）
```

## 快速开始

启动本地打分服务：

```powershell
python -X utf8 scripts/serve_demo.py --port 8000
```

接口说明见 `docs/前端对接说明.md`。

## 数据准备

`data/` 未纳入版本库（体积过大，且为公开/可复现数据）。需要时按下面步骤准备：

1. 下载 Polyvore disjoint 数据放入 `data/raw/polyvore/`
2. 提取 CLIP 特征：`python scripts/extract_features.py --split train`
3. 训练并评估：`python scripts/train_eval_final.py`

## 环境

建议使用虚拟环境（`python -m venv .venv`）。主要依赖：`torch`、`numpy`、`pandas`、`pyarrow`、`transformers`（CLIP）。
