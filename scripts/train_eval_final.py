# -*- coding: utf-8 -*-
"""
================================================================================
模块 F：训练 + 官方 valid 选模型 + 官方 test 最终评估  train_eval_final.py
================================================================================

用途
----
在 train / valid / test 三个划分的 CLIP 特征都具备后，做**最终**搭配兼容性基线评估：

    * 训练：只用 train 特征 + 官方 compatibility_train.txt；
    * 选模型/早停：只用 valid 特征 + 官方 compatibility_valid.txt（按 CP-AUC 选最佳 epoch）；
    * 最终评估：在最佳模型上，用 test 特征 + 官方 compatibility_test.txt 计算 CP-AUC，
      用 fill_in_blank_test.json 计算 FITB Top-1；
    * 保存最佳权重、训练日志、每轮指标、配置与随机种子。

严格防泄漏约定
--------------
    * 模型选择（最佳 epoch）只由 valid 决定，**绝不看 test**；
    * test 只做一次最终评估，不参与调参。

已核实的解析规则（以实际数据为准）：
    * compatibility token "{set_id}_{index}" 先拆 set_id 与 index，再经搭配 JSON 查到
      真实 item_id，最后查特征行；负样本是随机多 set_id 拼成的“假搭配”，token 需独立解析；
    * FITB 正确答案**不一定是 answers[0]**，必须按 (set_id, blank_position) 匹配唯一候选。

运行前先决条件
--------------
    需要三个划分的特征都已提取：
        data/processed/clip_features/           (train, 71967)
        data/processed/clip_features_valid/     (valid, 14657)
        data/processed/clip_features_test/      (test,  70035)  <- 需先提取
    test 特征提取命令（在项目 .venv 环境执行）：
        python -X utf8 scripts/extract_features.py --split test --smoke-test --out-dir data/processed/clip_features_test
        python -X utf8 scripts/extract_features.py --split test --full --batch-size 32 --num-workers 2 --out-dir data/processed/clip_features_test

运行方式
--------
  python -X utf8 scripts/train_eval_final.py
  python -X utf8 scripts/train_eval_final.py --epochs 100 --lr 1e-3 --hidden 256 --batch-size 512

输出（outputs/baseline_v1_final/）
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
DEFAULT_TEST_FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "clip_features_test"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_v1_final"


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
    return index_map.get(si[0], {}).get(si[1])


def build_feature_index(feat_dir: Path):
    feat = np.load(feat_dir / "features.npy")
    item_ids = load_json(feat_dir / "item_ids.json")
    if feat.shape[0] != len(item_ids):
        raise ValueError(f"特征行数 {feat.shape[0]} 与 item_ids.json 数量 {len(item_ids)} 不一致")
    return feat, {str(iid): r for r, iid in enumerate(item_ids)}


# ------------------------------------------------------------------------------
# 解析
# ------------------------------------------------------------------------------
def parse_compatibility(txt_path: Path, index_map, id_to_row, feat):
    X, y = [], []
    stats = Counter()
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.split()
        if not parts:
            continue
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
        y.append(1 if parts[0] == "1" else 0)
    if not X:
        raise ValueError(f"未解析出任何样本: {txt_path}")
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"), dict(stats)


def parse_fitb(fitb_path: Path, index_map, id_to_row, feat):
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
# 训练（valid 早停）
# ------------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_valid, y_valid, args, log_lines):
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
    t0 = time.time()

    print(f"\n开始训练：train={n} valid={len(Xv)}，epochs≤{args.epochs}，lr={args.lr}，"
          f"hidden={args.hidden}，batch={args.batch_size}，seed={args.seed}")
    log_lines.append("epoch\ttrain_loss\tvalid_auc\tpatience_left")

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

    return {
        "best_epoch": best_epoch,
        "best_valid_auc": float(best_auc),
        "elapsed_sec": round(time.time() - t0, 1),
        "history": history,
        "best_state": best_state,
    }


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="训练 + valid 选模型 + test 最终评估")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--train-feat-dir", type=str, default=str(DEFAULT_TRAIN_FEAT_DIR))
    parser.add_argument("--valid-feat-dir", type=str, default=str(DEFAULT_VALID_FEAT_DIR))
    parser.add_argument("--test-feat-dir", type=str, default=str(DEFAULT_TEST_FEAT_DIR))
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
    train_feat_dir = Path(args.train_feat_dir)
    valid_feat_dir = Path(args.valid_feat_dir)
    test_feat_dir = Path(args.test_feat_dir)

    if not _HAS_TORCH:
        print("!! 未安装 torch（pip install torch）")
        raise SystemExit(1)

    # test 特征前置检查
    if not (test_feat_dir / "features.npy").exists() or not (test_feat_dir / "item_ids.json").exists():
        print("!! 未找到 test 特征，请先提取：")
        print("   python -X utf8 scripts/extract_features.py --split test --smoke-test --out-dir data/processed/clip_features_test")
        print("   python -X utf8 scripts/extract_features.py --split test --full --batch-size 32 --num-workers 2 --out-dir data/processed/clip_features_test")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        import random as _r
        _r.seed(args.seed)
    except Exception:
        pass

    print("=" * 70)
    print("训练 + valid 选模型 + test 最终评估（均值池化 + MLP）")
    print("=" * 70)

    feat_tr, id2row_tr = build_feature_index(train_feat_dir)
    feat_va, id2row_va = build_feature_index(valid_feat_dir)
    feat_te, id2row_te = build_feature_index(test_feat_dir)
    print(f"[1] 特征：train={feat_tr.shape} valid={feat_va.shape} test={feat_te.shape}")

    im_tr = build_index_map(load_json(root / "disjoint" / "train.json"))
    im_va = build_index_map(load_json(root / "disjoint" / "valid.json"))
    im_te = build_index_map(load_json(root / "disjoint" / "test.json"))

    X_train, y_train, st_tr = parse_compatibility(root / "disjoint" / "compatibility_train.txt", im_tr, id2row_tr, feat_tr)
    X_valid, y_valid, st_va = parse_compatibility(root / "disjoint" / "compatibility_valid.txt", im_va, id2row_va, feat_va)
    X_test, y_test, st_te = parse_compatibility(root / "disjoint" / "compatibility_test.txt", im_te, id2row_te, feat_te)
    print(f"[2] compatibility: train={X_train.shape} 标签={Counter(y_train.tolist())} 解析={st_tr}")
    print(f"    valid={X_valid.shape} 标签={Counter(y_valid.tolist())} 解析={st_va}")
    print(f"    test ={X_test.shape} 标签={Counter(y_test.tolist())} 解析={st_te}")

    q_sum, q_n, cand, correct, fstats = parse_fitb(
        root / "disjoint" / "fill_in_blank_test.json", im_te, id2row_te, feat_te)
    print(f"[3] test FITB: 题目={fstats['num_questions']} 无匹配={fstats['no_match']} 多匹配={fstats['multi_match']}")
    print(f"    有限值检查: train={np.isfinite(X_train).all()} valid={np.isfinite(X_valid).all()} test={np.isfinite(X_test).all()}")

    model = MeanPoolMLP(in_dim=X_train.shape[1], hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[4] 模型参数量={n_params}")

    log_lines = []
    result = train_model(model, X_train, y_train, X_valid, y_valid, args, log_lines)

    # 最佳模型（由 valid 选定）
    best_model = MeanPoolMLP(in_dim=X_train.shape[1], hidden=args.hidden)
    best_model.load_state_dict(result["best_state"])
    best_model.eval()

    # 最终评估：test
    with torch.no_grad():
        test_logits = best_model(torch.from_numpy(X_test)).numpy()
    test_auc = roc_auc(y_test, test_logits)
    test_preds = fitb_top1(best_model, q_sum, q_n, cand)
    test_fitb = float((test_preds == correct).mean())

    # 复算 valid（供报告一致性确认）
    with torch.no_grad():
        valid_logits = best_model(torch.from_numpy(X_valid)).numpy()
    valid_auc_final = roc_auc(y_valid, valid_logits)

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
        "model_selection": "best_epoch 由官方 valid 的 CP-AUC 决定，test 不参与调参",
        "data": {
            "train_samples": int(len(X_train)),
            "valid_samples": int(len(X_valid)),
            "test_samples": int(len(X_test)),
            "train_label_dist": Counter(y_train.tolist()),
            "valid_label_dist": Counter(y_valid.tolist()),
            "test_label_dist": Counter(y_test.tolist()),
            "train_parse_stats": st_tr,
            "valid_parse_stats": st_va,
            "test_parse_stats": st_te,
            "test_fitb_questions": fstats["num_questions"],
        },
    }
    metrics = {
        "best_epoch": result["best_epoch"],
        "best_valid_auc": result["best_valid_auc"],       # 用于选模型
        "final_valid_auc": valid_auc_final,               # 最佳模型复算
        "test_auc": test_auc,                             # 最终 CP-AUC
        "test_fitb_top1": test_fitb,                      # 最终 FITB
        "elapsed_sec": result["elapsed_sec"],
        "history": result["history"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    np.savez(out_dir / "prepared_data.npz",
             X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
             X_test=X_test, y_test=y_test,
             fitb_qsum=q_sum, fitb_qn=q_n, fitb_cand=cand, fitb_correct=correct)

    print("\n" + "=" * 70)
    print("[5] 最终结果（官方 test）")
    print("=" * 70)
    print(f"  模型选择依据：官方 valid CP-AUC = {result['best_valid_auc']:.4f}（epoch {result['best_epoch']}）")
    print(f"  官方 valid CP-AUC（最佳模型复算）: {valid_auc_final:.4f}")
    print(f"  官方 test  CP-AUC: {test_auc:.4f}")
    print(f"  官方 test  FITB Top-1: {test_fitb:.4f}")
    print(f"  输出目录: {out_dir}")
    for f in ["best_model.pt", "config.json", "metrics.json", "training_log.txt", "prepared_data.npz"]:
        if (out_dir / f).exists():
            print(f"    - {f}")


if __name__ == "__main__":
    main()
