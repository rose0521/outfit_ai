# -*- coding: utf-8 -*-
"""
================================================================================
模块 D：第一版搭配兼容性基线模型  train_baseline.py
================================================================================

用途
----
基于已提取的 CLIP 特征（均值池化 + MLP）训练第一版搭配兼容性基线：

    * 任务一：根据 item_ids.json 建立 item_id -> 特征行号 映射（禁止按 Parquet 行号索引）；
    * 任务二：解析 compatibility_*.txt 正负样本（token "{set_id}_{index}" 先拆 set_id
      与 index，再经搭配 JSON 查真实 item_id，最后查特征行）；已核实负样本行是
      随机多 set_id 拼成的“假搭配”，每个 token 必须独立解析；
    * 任务三：均值池化得到整套表示，再用 512 -> 256 -> 1 的 MLP 输出兼容分数，
      BCE 训练，CLIP 特征冻结；
    * 任务四：训练集只更新参数，验证集用于选模型/早停，计算 CP-AUC 与 FITB Top-1；
    * 任务五：保存最佳权重、训练日志、每轮指标、配置与随机种子。

关于验证集的重要说明（务必阅读）
--------------------------------
当前 CLIP 特征**只提取了 train 划分**（71967 个商品 = train.parquet 全集）。
官方 valid/test 的商品大多不在 train 中（跨划分仅 3781 个重叠），因此：

    * 官方 valid/test 的 CP-AUC 与 FITB 无法用现有特征正确计算（本脚本会输出
      覆盖缺口诊断：实测 valid 有 22182 个 token 无特征、1255 行整行为空）；
    * 本脚本退而采用 **train 内部按 set_id 留出的验证集**（默认 15%）做模型选择，
      正样本为真实搭配，负样本按官方风格（随机抽取分区内商品拼成假搭配）生成；
    * 该内部验证集**不是**官方 valid，仅供第一版选模型/早停；要得到官方 valid/test
      指标，需先对 validation.parquet / test.parquet 提取特征（后续步骤，不属本阶段）。

设计原则
--------
    * 只读 CLIP 特征，绝不修改/覆盖；
    * 不读 Parquet 图片；
    * 只依赖 numpy + torch（CPU）；
    * 固定随机种子，可复现；
    * 均值池化在 numpy 侧预计算，训练只跑 MLP，CPU 秒级完成。

运行方式
--------
  python -X utf8 scripts/train_baseline.py
  python -X utf8 scripts/train_baseline.py --epochs 100 --lr 1e-3 --hidden 256 --batch-size 512
  python -X utf8 scripts/train_baseline.py --out-dir outputs/baseline_v1 --seed 42 --holdout-frac 0.15

输出（outputs/baseline_v1/）
    * best_model.pt            最佳模型权重（state_dict）
    * config.json              模型配置、超参、随机种子、数据规模、验证集构成说明
    * metrics.json             每轮指标 + 最佳 CP-AUC / FITB Top-1
    * training_log.txt         人类可读训练日志
    * prepared_data.npz        预处理数据（供后续迭代复用）

依赖
----
  pip install numpy torch
================================================================================
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ------------------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "polyvore"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_v1"


# ------------------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def token_to_set_index(token):
    """token "{set_id}_{index}" -> (set_id, index)；失败返回 None。"""
    if "_" not in token:
        return None
    set_id, idx_s = token.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        return None
    return set_id, idx


def build_index_map(outfits):
    """由搭配 JSON 建 set_id -> {index: item_id}。"""
    m = {}
    for o in outfits:
        inner = {}
        for it in o.get("items", []):
            if isinstance(it, dict) and "index" in it and "item_id" in it:
                inner[it["index"]] = it["item_id"]
        if "set_id" in o:
            m[o["set_id"]] = inner
    return m


def resolve_token(token, index_map):
    """token -> item_id；失败返回 None。"""
    si = token_to_set_index(token)
    if si is None:
        return None
    set_id, idx = si
    inner = index_map.get(set_id)
    if inner is None:
        return None
    return inner.get(idx)


# ------------------------------------------------------------------------------
# 任务一：特征索引
# ------------------------------------------------------------------------------
def build_feature_index(feat_dir: Path):
    """读取 item_ids.json 建 item_id -> 特征行号。严禁按 Parquet 行号索引。"""
    feat = np.load(feat_dir / "features.npy")
    item_ids = load_json(feat_dir / "item_ids.json")
    if feat.shape[0] != len(item_ids):
        raise ValueError(f"特征行数 {feat.shape[0]} 与 item_ids.json 数量 {len(item_ids)} 不一致")
    id_to_row = {str(iid): r for r, iid in enumerate(item_ids)}
    return feat, id_to_row


# ------------------------------------------------------------------------------
# 任务二：解析官方 compatibility（诊断 + 覆盖检查）
# ------------------------------------------------------------------------------
def parse_compatibility(txt_path: Path, index_map, id_to_row, feat):
    """
    解析 compatibility_{split}.txt，返回 (X, y, stats)。
    X 为均值池化后的整套向量 (n,512)，y 为标签 (n,) 0/1。
    每个 token 独立解析：token -> (set_id, index) -> item_id -> 特征行。
    stats 记录 unresolved_token / no_feature / empty_line 数量。
    """
    X, y = [], []
    stats = Counter()
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.split()
        if not parts:
            continue
        label = parts[0]
        tokens = parts[1:]
        rows = []
        for t in tokens:
            item_id = resolve_token(t, index_map)
            if item_id is None:
                stats["unresolved_token"] += 1
                continue
            row = id_to_row.get(str(item_id))
            if row is None:
                stats["no_feature"] += 1
                continue
            rows.append(row)
        if not rows:
            stats["empty_line"] += 1
            continue
        X.append(feat[rows].mean(axis=0))
        y.append(1 if label == "1" else 0)
    if not X:
        return None, None, dict(stats)
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="float32")
    return X, y, dict(stats)


def diagnose_official_valid(feat, id_to_row, index_map, root):
    """
    诊断官方 valid 集在「仅 train 特征」下的覆盖率，返回报告 dict。
    用于明确告知：官方 valid 尚不可评估。
    """
    txt = root / "disjoint" / "compatibility_valid.txt"
    total_lines = sum(1 for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.split())
    _, _, stats = parse_compatibility(txt, index_map, id_to_row, feat)
    return {
        "compatibility_valid_total_lines": total_lines,
        "unresolved_token": stats.get("unresolved_token", 0),
        "no_feature": stats.get("no_feature", 0),
        "empty_line": stats.get("empty_line", 0),
        "usable_lines": total_lines - stats.get("empty_line", 0),
        "reason": "CLIP 特征仅覆盖 train 划分商品，官方 valid 商品大多无特征",
    }


# ------------------------------------------------------------------------------
# train 内部按 set_id 留出验证集
# ------------------------------------------------------------------------------
def split_and_build_holdout(outfits, feat, id_to_row, seed, holdout_frac):
    """
    把 train 的搭配按 set_id 划分为 train_part / val_part，构造：
        X_train/y_train/X_val/y_val（均值池化向量 + 0/1 标签）
        fitb_val（q_sum/q_n/cand/correct）
    正样本 = 真实搭配；负样本 = 分区内随机抽取商品拼成的“假搭配”（官方风格）。
    返回 dict。
    """
    rng = np.random.default_rng(seed)

    set_ids = sorted({o["set_id"] for o in outfits if "set_id" in o})
    rng.shuffle(set_ids)
    n_val = max(1, int(round(len(set_ids) * holdout_frac)))
    val_sets = set(set_ids[:n_val])

    train_poses = []   # (mean_vec, rows)
    val_poses = []     # (mean_vec, rows)
    val_outfits = []   # (set_id, [rows]) 用于 FITB
    train_pool, val_pool = [], []

    for o in outfits:
        sid = o.get("set_id")
        rows = [id_to_row[str(it["item_id"])] for it in o.get("items", [])
                if str(it.get("item_id")) in id_to_row]
        if not rows:
            continue
        vec = feat[rows].mean(axis=0).astype("float32")
        if sid in val_sets:
            val_poses.append((vec, rows))
            val_outfits.append((sid, rows))
            val_pool.extend(rows)
        else:
            train_poses.append((vec, rows))
            train_pool.extend(rows)

    train_pool = list(set(train_pool))
    val_pool = list(set(val_pool))

    def make_negatives(poses, pool):
        negs = []
        for _, rows in poses:
            k = len(rows)
            if k <= len(pool):
                sampled = rng.choice(pool, size=k, replace=False)
            else:
                sampled = rng.choice(pool, size=k, replace=True)
            negs.append(feat[sampled].mean(axis=0).astype("float32"))
        return np.asarray(negs, dtype="float32")

    X_train_pos = np.asarray([p[0] for p in train_poses], dtype="float32")
    X_val_pos = np.asarray([p[0] for p in val_poses], dtype="float32")
    X_train_neg = make_negatives(train_poses, train_pool)
    X_val_neg = make_negatives(val_poses, val_pool)

    X_train = np.concatenate([X_train_pos, X_train_neg], axis=0)
    y_train = np.concatenate([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))]).astype("float32")
    X_val = np.concatenate([X_val_pos, X_val_neg], axis=0)
    y_val = np.concatenate([np.ones(len(X_val_pos)), np.zeros(len(X_val_neg))]).astype("float32")

    # 打乱训练集（保持确定性）
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # 构造 FITB（验证集）：每套 val 搭配随机挖一个空，3 个干扰项来自 val_pool
    q_sum, q_n, cand, correct = [], [], [], []
    for sid, rows in val_outfits:
        k = len(rows)
        b = int(rng.integers(0, k))          # 0-based 挖空位置
        question_rows = [r for i, r in enumerate(rows) if i != b]
        correct_row = rows[b]
        excl = set(question_rows) | {correct_row}
        cand_pool = [r for r in val_pool if r not in excl]
        if len(cand_pool) < 3:
            cand_pool = [r for r in val_pool if r != correct_row]
        if len(cand_pool) >= 3:
            dist = rng.choice(cand_pool, size=3, replace=False)
        else:
            dist = rng.choice(val_pool, size=3, replace=True)
        cand_rows = [correct_row] + list(dist)
        rng.shuffle(cand_rows)
        ci = cand_rows.index(correct_row)
        q_sum.append(feat[question_rows].sum(axis=0).astype("float32"))
        q_n.append(len(question_rows))
        cand.append(np.stack([feat[r] for r in cand_rows]).astype("float32"))
        correct.append(ci)

    fitb_val = {
        "q_sum": np.asarray(q_sum, dtype="float32"),
        "q_n": np.asarray(q_n, dtype="float32"),
        "cand": np.asarray(cand, dtype="float32"),
        "correct": np.asarray(correct, dtype="int64"),
        "num_questions": len(q_sum),
    }

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "fitb_val": fitb_val,
        "n_train_outfits": len(train_poses),
        "n_val_outfits": len(val_poses),
        "train_pool_size": len(train_pool),
        "val_pool_size": len(val_pool),
        "val_sets": val_sets,
    }


# ------------------------------------------------------------------------------
# 模型
# ------------------------------------------------------------------------------
class MeanPoolMLP(nn.Module):
    """均值池化后的整套向量 -> MLP -> 兼容性分数（logit）。"""

    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------------------
# 指标
# ------------------------------------------------------------------------------
def roc_auc(y_true, y_score):
    """手动计算二分类 AUC（Mann-Whitney U），避免 sklearn 依赖。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        less = np.searchsorted(neg, p, side="left")
        eq = np.searchsorted(neg, p, side="right") - less
        wins += less + 0.5 * eq
    return wins / (len(pos) * len(neg))


def fitb_top1(model, fitb_val):
    """FITB Top-1 准确率。整套 = (问题项之和 + 候选) / (问题项数 + 1)，打分取最高。"""
    model.eval()
    q_sum, q_n, cand = fitb_val["q_sum"], fitb_val["q_n"], fitb_val["cand"]
    m = q_sum.shape[0]
    preds = np.zeros(m, dtype="int64")
    with torch.no_grad():
        for i in range(m):
            n = q_n[i]
            z = (q_sum[i][None, :] + cand[i]) / (n + 1)
            scores = model(torch.from_numpy(z)).numpy()
            preds[i] = int(np.argmax(scores))
    return preds


# ------------------------------------------------------------------------------
# 训练
# ------------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val, fitb_val, args, out_dir, log_lines):
    device = "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)

    n = len(Xt)
    best_auc = -1.0
    best_state = None
    best_epoch = -1
    best_fitb = -1.0
    patience_left = args.patience
    history = []

    print(f"\n开始训练：train={n} val={len(Xv)}，epochs≤{args.epochs}，lr={args.lr}，"
          f"hidden={args.hidden}，batch={args.batch_size}，seed={args.seed}")
    log_lines.append("epoch\ttrain_loss\tval_auc\tfitb_top1\tpatience_left")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        nb = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            nb += len(idx)
        train_loss = total_loss / nb if nb else float("nan")

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).numpy()
        val_auc = roc_auc(y_val, val_logits)
        preds = fitb_top1(model, fitb_val)
        fitb_acc = float((preds == fitb_val["correct"]).mean()) if fitb_val["num_questions"] else float("nan")

        if not (np.isfinite(train_loss) and np.isfinite(val_auc)):
            print(f"!! epoch {epoch}: 出现非有限值，停止。train_loss={train_loss} val_auc={val_auc}")
            break

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_auc),
            "fitb_top1": float(fitb_acc),
        })
        print(f"epoch {epoch:>3d}  train_loss={train_loss:.4f}  "
              f"val_auc={val_auc:.4f}  fitb_top1={fitb_acc:.4f}  "
              f"best_auc={max(best_auc, val_auc):.4f}")
        log_lines.append(f"{epoch}\t{train_loss:.6f}\t{val_auc:.6f}\t{fitb_acc:.6f}\t{patience_left}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_fitb = fitb_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"早停：验证 AUC 连续 {args.patience} 轮未提升，停止于 epoch {epoch}。")
                break

    elapsed = time.time() - t0
    print(f"\n训练结束，耗时 {elapsed:.1f}s。最佳内部验证 AUC={best_auc:.4f}（epoch {best_epoch}），"
          f"该 epoch 的 FITB Top-1={best_fitb:.4f}")
    return {
        "best_epoch": best_epoch,
        "best_valid_auc": float(best_auc),
        "best_fitb_top1": float(best_fitb),
        "elapsed_sec": round(elapsed, 1),
        "history": history,
        "best_state": best_state,
    }


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="第一版搭配兼容性基线（均值池化 + MLP）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--feature-dir", type=str, default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--holdout-frac", type=float, default=0.15,
                        help="train 内部留出验证集的比例（默认 0.15）")
    args = parser.parse_args()

    root = Path(args.root)
    feat_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)

    if not _HAS_TORCH:
        print("!! 未安装 torch，无法训练（pip install torch）")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        import random as _random
        _random.seed(args.seed)
    except Exception:
        pass

    print("=" * 70)
    print("第一版搭配兼容性基线（均值池化 + MLP）")
    print("=" * 70)
    print(f"特征目录: {feat_dir}")
    print(f"输出目录: {out_dir}")
    print(f"随机种子: {args.seed}  验证集留出比例: {args.holdout_frac}")

    # 任务一：特征索引
    print("\n[1] 建立 item_id -> 特征行号 索引 ...")
    feat, id_to_row = build_feature_index(feat_dir)
    print(f"    特征矩阵 shape={feat.shape} dtype={feat.dtype}，索引条数={len(id_to_row)}")

    # 读取 train 搭配 JSON
    train_outfits = load_json(root / "disjoint" / "train.json")
    train_index_map = build_index_map(train_outfits)
    print(f"    train 搭配: {len(train_outfits)} 套")

    # 任务二：解析官方 compatibility（诊断）
    print("\n[2] 解析官方 compatibility_train.txt（诊断）...")
    X_off, y_off, stats_off = parse_compatibility(
        root / "disjoint" / "compatibility_train.txt", train_index_map, id_to_row, feat)
    print(f"    官方 train 样本: X={X_off.shape} y={y_off.shape} 标签分布={Counter(y_off.tolist())} 解析统计={stats_off}")
    print("    （确认：负样本为随机多 set_id 拼成的假搭配，token 需独立解析）")

    # 官方 valid 覆盖诊断
    print("\n[3] 官方 valid 覆盖率诊断 ...")
    valid_index_map = build_index_map(load_json(root / "disjoint" / "valid.json"))
    diag = diagnose_official_valid(feat, id_to_row, valid_index_map, root)
    print(f"    官方 valid 总行数={diag['compatibility_valid_total_lines']}，"
          f"无特征 token={diag['no_feature']}，整行为空={diag['empty_line']}，"
          f"可用行={diag['usable_lines']}")
    print(f"    !! 结论：{diag['reason']}。因此本脚本改用 train 内部留出验证集。")

    # 内部留出验证集
    print(f"\n[4] 构造 train 内部留出验证集（按 set_id，{args.holdout_frac:.0%}）...")
    data = split_and_build_holdout(train_outfits, feat, id_to_row, args.seed, args.holdout_frac)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    fitb_val = data["fitb_val"]
    print(f"    train 搭配数={data['n_train_outfits']}，val 搭配数={data['n_val_outfits']}")
    print(f"    train 商品池={data['train_pool_size']}，val 商品池={data['val_pool_size']}")
    print(f"    训练样本={len(X_train)}（正={int(y_train.sum())} 负={int(len(y_train)-y_train.sum())}）")
    print(f"    验证样本={len(X_val)}（正={int(y_val.sum())} 负={int(len(y_val)-y_val.sum())}）")
    print(f"    FITB 题目={fitb_val['num_questions']}")
    print(f"    有限值检查: train={np.isfinite(X_train).all()} val={np.isfinite(X_val).all()}")

    # 模型与训练
    print("\n[5] 构建模型并训练 ...")
    model = MeanPoolMLP(in_dim=X_train.shape[1], hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数量={n_params}")

    log_lines = []
    result = train_model(model, X_train, y_train, X_val, y_val, fitb_val, args, out_dir, log_lines)

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    if result["best_state"] is not None:
        torch.save(result["best_state"], out_dir / "best_model.pt")

    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "model": "MeanPoolMLP",
        "arch": {"in_dim": int(X_train.shape[1]), "hidden": args.hidden, "out_dim": 1},
        "pooling": "mean (L2-normalized CLIP features, frozen)",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "holdout_frac": args.holdout_frac,
        "n_params": n_params,
        "validation_scheme": "train 内部按 set_id 留出验证集；正样本=真实搭配，负样本=官方风格随机假搭配",
        "official_valid_blocked_reason": diag["reason"],
        "official_valid_coverage": diag,
        "data": {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "train_label_dist": Counter(y_train.tolist()),
            "val_label_dist": Counter(y_val.tolist()),
            "n_train_outfits": data["n_train_outfits"],
            "n_val_outfits": data["n_val_outfits"],
            "official_train_parse_stats": stats_off,
            "fitb_questions": fitb_val["num_questions"],
        },
    }
    metrics = {
        "best_epoch": result["best_epoch"],
        "best_valid_auc": result["best_valid_auc"],
        "best_fitb_top1": result["best_fitb_top1"],
        "elapsed_sec": result["elapsed_sec"],
        "history": result["history"],
    }

    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    np.savez(
        out_dir / "prepared_data.npz",
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        fitb_qsum=fitb_val["q_sum"], fitb_qn=fitb_val["q_n"],
        fitb_cand=fitb_val["cand"], fitb_correct=fitb_val["correct"],
    )

    print("\n" + "=" * 70)
    print("[6] 结果汇总")
    print("=" * 70)
    print(f"  最佳（内部）验证 CP-AUC: {result['best_valid_auc']:.4f}（epoch {result['best_epoch']}）")
    if result["best_fitb_top1"] >= 0:
        print(f"  最佳 epoch 的 FITB Top-1: {result['best_fitb_top1']:.4f}")
    print(f"  （注意：以上为 train 内部留出验证集，非官方 valid；官方 valid 需先提取其商品特征。）")
    print(f"  输出目录: {out_dir}")
    for f in ["best_model.pt", "config.json", "metrics.json", "training_log.txt", "prepared_data.npz"]:
        p = out_dir / f
        if p.exists():
            print(f"    - {f}")


if __name__ == "__main__":
    main()
