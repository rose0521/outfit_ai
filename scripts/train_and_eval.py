# -*- coding: utf-8 -*-
"""
================================================================================
模块 E：训练 + 官方 valid 评估  train_and_eval.py
================================================================================

用途
----
在 train / valid 两个划分的 CLIP 特征都已具备后，做**正式**的搭配兼容性基线：

    * 训练：只用 train 划分的 CLIP 特征与官方 compatibility_train.txt（正负样本）；
    * 验证/选模型：只用 valid 划分特征与官方 compatibility_valid.txt，计算 CP-AUC；
    * 评估：在最佳模型上计算官方 valid 的 CP-AUC 与 FITB Top-1 准确率；
    * 保存最佳权重、训练日志、每轮指标、配置与随机种子。

与 train_baseline.py（train 内部留出验证）的区别
------------------------------------------------
本脚本使用**官方 valid 划分**做验证与评估，结果可直接作为正式指标（当前已具备
valid 特征 data/processed/clip_features_valid/）。两个脚本不互相依赖，可独立运行。

已核实的关键规则（以实际数据为准）：
    * compatibility token "{set_id}_{index}"：先拆 set_id 与 index，再经搭配 JSON
      查到真实 item_id，最后查特征行；负样本是随机多 set_id 拼成的“假搭配”，
      每个 token 必须独立解析；
    * valid 集所有 token 都能用 valid 特征单独解析（无 train_only、无 unresolved）；
    * FITB 正确答案**不一定是 answers[0]**，必须按 (set_id, blank_position) 匹配
      唯一候选（实测 answers[0] 仅为正确答案的概率约 25%）。

模型
----
均值池化（整套 = 各单品 512 维特征的平均，CLIP 冻结）+ MLP(512→256→1) + BCE。

运行方式
--------
  python -X utf8 scripts/train_and_eval.py
  python -X utf8 scripts/train_and_eval.py --epochs 100 --lr 1e-3 --hidden 256 --batch-size 512

输出（outputs/baseline_v1_official/）
    * best_model.pt / config.json / metrics.json / training_log.txt / prepared_data.npz

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
DEFAULT_TRAIN_FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features"
DEFAULT_VALID_FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features_valid"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_v1_official"


# ------------------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def token_to_set_index(token):
    if "_" not in token:
        return None
    set_id, idx_s = token.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        return None
    return set_id, idx


def build_index_map(outfits):
    m = {}
    for o in outfits:
        m[o["set_id"]] = {it["index"]: it["item_id"]
                          for it in o.get("items", []) if isinstance(it, dict)}
    return m


def resolve_token(token, index_map):
    si = token_to_set_index(token)
    if si is None:
        return None
    set_id, idx = si
    return index_map.get(set_id, {}).get(idx)


# ------------------------------------------------------------------------------
# 特征索引
# ------------------------------------------------------------------------------
def build_feature_index(feat_dir: Path):
    feat = np.load(feat_dir / "features.npy")
    item_ids = load_json(feat_dir / "item_ids.json")
    if feat.shape[0] != len(item_ids):
        raise ValueError(f"特征行数 {feat.shape[0]} 与 item_ids.json 数量 {len(item_ids)} 不一致")
    id_to_row = {str(iid): r for r, iid in enumerate(item_ids)}
    return feat, id_to_row


# ------------------------------------------------------------------------------
# 解析 compatibility -> (X, y)
# ------------------------------------------------------------------------------
def parse_compatibility(txt_path: Path, index_map, id_to_row, feat):
    X, y = [], []
    stats = Counter()
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.split()
        if not parts:
            continue
        label = parts[0]
        rows = []
        for t in parts[1:]:
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
        raise ValueError(f"未解析出任何样本: {txt_path}")
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"), dict(stats)


# ------------------------------------------------------------------------------
# 解析 FITB
# ------------------------------------------------------------------------------
def parse_fitb(fitb_path: Path, index_map, id_to_row, feat):
    """
    返回 (q_sum, q_n, cand, correct, stats)：
        q_sum   (m,512) 问题项特征之和；q_n (m,) 问题项数量；
        cand    (m,4,512) 4 候选特征；correct (m,) 正确答案下标。
    正确答案按 (set_id, blank_position) 匹配，绝不假定为 answers[0]。
    """
    data = load_json(fitb_path)
    if not isinstance(data, list):
        raise ValueError(f"FITB 顶层类型不是 list，而是 {type(data).__name__}")
    q_sum, q_n, cand, correct = [], [], [], []
    n_no_match = n_multi = 0
    for q in data:
        question = q.get("question", [])
        answers = q.get("answers", [])
        bp = q.get("blank_position")
        if not question or not answers:
            continue
        qset = token_to_set_index(question[0])
        qset = qset[0] if qset else None
        rows = []
        for t in question:
            iid = resolve_token(t, index_map)
            row = id_to_row.get(str(iid)) if iid is not None else None
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        q_sum.append(feat[rows].sum(axis=0))
        q_n.append(len(rows))
        cand_rows, ci = [], None
        for c_idx, t in enumerate(answers):
            iid = resolve_token(t, index_map)
            row = id_to_row.get(str(iid)) if iid is not None else None
            cand_rows.append(feat[row] if row is not None else np.zeros(feat.shape[1], dtype="float32"))
            si = token_to_set_index(t)
            if si is not None and qset is not None and si[0] == qset and si[1] == bp:
                if ci is None:
                    ci = c_idx
                else:
                    n_multi += 1
        if ci is None:
            n_no_match += 1
            ci = 0
        cand.append(np.stack(cand_rows, axis=0))
        correct.append(ci)
    if not q_sum:
        raise ValueError("FITB 未解析出任何题目")
    return (np.asarray(q_sum, dtype="float32"),
            np.asarray(q_n, dtype="float32"),
            np.asarray(cand, dtype="float32"),
            np.asarray(correct, dtype="int64"),
            {"num_questions": len(q_sum), "no_match": n_no_match, "multi_match": n_multi})


# ------------------------------------------------------------------------------
# 模型与指标
# ------------------------------------------------------------------------------
class MeanPoolMLP(nn.Module):
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def roc_auc(y_true, y_score):
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


def fitb_top1(model, q_sum, q_n, cand):
    model.eval()
    m = q_sum.shape[0]
    preds = np.zeros(m, dtype="int64")
    with torch.no_grad():
        for i in range(m):
            n = q_n[i]
            z = (q_sum[i][None, :] + cand[i]) / (n + 1)
            preds[i] = int(np.argmax(model(torch.from_numpy(z)).numpy()))
    return preds


# ------------------------------------------------------------------------------
# 训练
# ------------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_valid, y_valid, args, out_dir, log_lines):
    model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_valid)

    n = len(Xt)
    best_auc, best_state, best_epoch = -1.0, None, -1
    patience_left = args.patience
    history = []

    print(f"\n开始训练：train={n} valid={len(Xv)}，epochs≤{args.epochs}，lr={args.lr}，"
          f"hidden={args.hidden}，batch={args.batch_size}，seed={args.seed}")
    log_lines.append("epoch\ttrain_loss\tvalid_auc\tpatience_left")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss, nb = 0.0, 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            nb += len(idx)
        train_loss = total_loss / nb if nb else float("nan")

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).numpy()
        val_auc = roc_auc(y_valid, val_logits)

        if not (np.isfinite(train_loss) and np.isfinite(val_auc)):
            print(f"!! epoch {epoch}: 出现非有限值，停止。")
            break

        history.append({"epoch": epoch, "train_loss": float(train_loss), "valid_auc": float(val_auc)})
        print(f"epoch {epoch:>3d}  train_loss={train_loss:.4f}  valid_auc={val_auc:.4f}  "
              f"best_auc={max(best_auc, val_auc):.4f}")
        log_lines.append(f"{epoch}\t{train_loss:.6f}\t{val_auc:.6f}\t{patience_left}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"早停：验证 AUC 连续 {args.patience} 轮未提升，停止于 epoch {epoch}。")
                break

    elapsed = time.time() - t0
    return {
        "best_epoch": best_epoch,
        "best_valid_auc": float(best_auc),
        "elapsed_sec": round(elapsed, 1),
        "history": history,
        "best_state": best_state,
    }


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="训练 + 官方 valid 评估（均值池化 + MLP）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--train-feat-dir", type=str, default=str(DEFAULT_TRAIN_FEAT_DIR))
    parser.add_argument("--valid-feat-dir", type=str, default=str(DEFAULT_VALID_FEAT_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if not _HAS_TORCH:
        print("!! 未安装 torch（pip install torch）")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        import random as _r
        _r.seed(args.seed)
    except Exception:
        pass

    print("=" * 70)
    print("训练 + 官方 valid 评估（均值池化 + MLP）")
    print("=" * 70)

    # 特征索引
    feat_tr, id_to_row_tr = build_feature_index(Path(args.train_feat_dir))
    feat_va, id_to_row_va = build_feature_index(Path(args.valid_feat_dir))
    print(f"[1] 特征：train={feat_tr.shape}，valid={feat_va.shape}")

    train_outfits = load_json(root / "disjoint" / "train.json")
    valid_outfits = load_json(root / "disjoint" / "valid.json")
    im_tr = build_index_map(train_outfits)
    im_va = build_index_map(valid_outfits)

    # 训练数据（官方 train）
    X_train, y_train, st_tr = parse_compatibility(
        root / "disjoint" / "compatibility_train.txt", im_tr, id_to_row_tr, feat_tr)
    print(f"[2] train compatibility: X={X_train.shape} y={y_train.shape} "
          f"标签分布={Counter(y_train.tolist())} 解析统计={st_tr}")

    # 验证数据（官方 valid）
    X_valid, y_valid, st_va = parse_compatibility(
        root / "disjoint" / "compatibility_valid.txt", im_va, id_to_row_va, feat_va)
    print(f"[3] valid compatibility: X={X_valid.shape} y={y_valid.shape} "
          f"标签分布={Counter(y_valid.tolist())} 解析统计={st_va}")

    # FITB（官方 valid）
    q_sum, q_n, cand, correct, fstats = parse_fitb(
        root / "disjoint" / "fill_in_blank_valid.json", im_va, id_to_row_va, feat_va)
    print(f"[4] valid FITB: 题目={fstats['num_questions']} "
          f"无匹配={fstats['no_match']} 多匹配={fstats['multi_match']}")
    print(f"    有限值检查: train={np.isfinite(X_train).all()} valid={np.isfinite(X_valid).all()}")

    # 模型与训练
    model = MeanPoolMLP(in_dim=X_train.shape[1], hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[5] 模型参数量={n_params}")

    log_lines = []
    result = train_model(model, X_train, y_train, X_valid, y_valid, args, out_dir, log_lines)

    # 最佳模型上的 FITB
    best_model = MeanPoolMLP(in_dim=X_train.shape[1], hidden=args.hidden)
    best_model.load_state_dict(result["best_state"])
    preds = fitb_top1(best_model, q_sum, q_n, cand)
    fitb_acc = float((preds == correct).mean())

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "n_params": n_params,
        "eval": "官方 valid 划分（CP-AUC + FITB Top-1）",
        "data": {
            "train_samples": int(len(X_train)),
            "valid_samples": int(len(X_valid)),
            "train_label_dist": Counter(y_train.tolist()),
            "valid_label_dist": Counter(y_valid.tolist()),
            "train_parse_stats": st_tr,
            "valid_parse_stats": st_va,
            "fitb_questions": fstats["num_questions"],
        },
    }
    metrics = {
        "best_epoch": result["best_epoch"],
        "best_valid_auc": result["best_valid_auc"],
        "fitb_top1": fitb_acc,
        "elapsed_sec": result["elapsed_sec"],
        "history": result["history"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    np.savez(out_dir / "prepared_data.npz",
             X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
             fitb_qsum=q_sum, fitb_qn=q_n, fitb_cand=cand, fitb_correct=correct)

    print("\n" + "=" * 70)
    print("[6] 结果汇总（官方 valid）")
    print("=" * 70)
    print(f"  最佳官方 valid CP-AUC: {result['best_valid_auc']:.4f}（epoch {result['best_epoch']}）")
    print(f"  最佳模型的 FITB Top-1: {fitb_acc:.4f}")
    print(f"  输出目录: {out_dir}")
    for f in ["best_model.pt", "config.json", "metrics.json", "training_log.txt", "prepared_data.npz"]:
        if (out_dir / f).exists():
            print(f"    - {f}")


if __name__ == "__main__":
    main()
