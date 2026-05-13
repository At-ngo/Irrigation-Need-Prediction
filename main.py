import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import f1_score

from data_loader import load_data
from feature_engineering import FE_REGISTRY, apply_fe
from train import run_experiment

RESULTS_PATH = "results/fe_results.csv"
os.makedirs("results",  exist_ok=True)
os.makedirs("outputs",  exist_ok=True)

TARGET    = "Irrigation_Need"
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
ID_MAP    = {v: k for k, v in LABEL_MAP.items()}   # 0→Low, 1→Medium, 2→High


# ── 1. Experiment loop ────────────────────────────────────────
def run_all_experiments(train_raw, cfg):
    all_results = []
    for config_name in FE_REGISTRY:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config_name}")
        result = run_experiment(train_raw, cfg, config_name, apply_fe)
        all_results.append(result)
        print(f"  => cv_mean={result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "fold_scores"}
        for r in all_results
    ]).sort_values("cv_mean", ascending=False).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved → {RESULTS_PATH}")
    print(results_df.to_string(index=False))
    return results_df


# ── 2. Train full data với best config, predict test ─────────
def train_full_and_predict(train_raw, test_raw, best_config, cfg,
                           n_splits=5, random_state=42):
    print(f"\n{'='*50}")
    print(f"Training FULL data with best config: {best_config}")
    print(f"{'='*50}")

    # Apply FE cho cả train và test
    train_fe = apply_fe(train_raw, best_config)
    test_fe  = apply_fe(test_raw,  best_config)

    id_col   = cfg["competition"].get("id_col", "id")

    # Cột drop: target + id
    drop_train = [c for c in [TARGET, id_col] if c in train_fe.columns]
    drop_test  = [c for c in [id_col]         if c in test_fe.columns]

    # Encode categorical — fit trên train, transform cả hai
    cat_cols = [c for c in train_fe.columns
                if train_fe[c].dtype in ["object", "category"]
                and c not in drop_train]

    train_enc = pd.get_dummies(train_fe.drop(columns=drop_train), columns=cat_cols)
    test_enc  = pd.get_dummies(test_fe.drop(columns=drop_test),   columns=cat_cols)

    # Align columns — test có thể thiếu 1 số dummy cols
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    feature_cols = train_enc.columns.tolist()
    y = train_fe[TARGET].map(LABEL_MAP)

    # ── K-fold: lưu OOF + test predictions ───────────────────
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    oof_preds  = np.zeros(len(train_enc))
    test_preds = np.zeros(len(test_enc))   # average qua các fold
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_enc, y)):
        X_tr,  X_val = train_enc.iloc[tr_idx],  train_enc.iloc[val_idx]
        y_tr,  y_val = y.iloc[tr_idx],          y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=1000,          # nhiều hơn lúc experiment
            class_weight="balanced",
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred  = model.predict(X_val)
        test_fold = model.predict(test_enc)

        oof_preds[val_idx] = val_pred
        test_preds        += test_fold / n_splits   # average

        score = f1_score(y_val, val_pred, average="macro")
        fold_scores.append(score)
        print(f"  Fold {fold+1}/{n_splits}  F1-macro={score:.4f}")

    oof_score = f1_score(y, oof_preds, average="macro")
    print(f"\n  OOF F1-macro (full train): {oof_score:.4f}")

    # test_preds là average của predict() → làm tròn về class gần nhất
    final_labels = np.round(test_preds).astype(int).clip(0, 2)
    return final_labels, oof_score


# ── 3. Xuất submission ────────────────────────────────────────
def make_submission(test_raw, final_labels, cfg, best_config, oof_score):
    id_col = cfg["competition"].get("id_col", "id")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname  = f"submission_{best_config}_{ts}.csv"
    fpath  = os.path.join("outputs", fname)

    sub = pd.DataFrame({
        id_col        : test_raw[id_col],
        TARGET        : [ID_MAP[i] for i in final_labels],
    })
    sub.to_csv(fpath, index=False)

    print(f"\n[Submission] Saved → {fpath}")
    print(f"[Submission] OOF F1-macro: {oof_score:.4f}")
    print(f"[Submission] Label distribution:")
    print(sub[TARGET].value_counts().to_string())
    print(sub.head(5).to_string(index=False))
    return fpath


# ── Main ──────────────────────────────────────────────────────
def main():
    with open("configs/config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_raw, test_raw, _ = load_data(cfg)

    # Bước 1: chạy tất cả FE experiments
    results_df = run_all_experiments(train_raw, cfg)

    # Bước 2-4: train full + xuất submission cho TẤT CẢ configs
    print(f"\n{'='*50}")
    print(f"Xuất submission cho tất cả {len(results_df)} configs...")

    for _, row in results_df.iterrows():
        config_name = row["fe_config"]
        print(f"\n>>> [{config_name}]  CV={row['cv_mean']:.4f}")
        final_labels, oof_score = train_full_and_predict(
            train_raw, test_raw, config_name, cfg
        )
        make_submission(test_raw, final_labels, cfg, config_name, oof_score)


if __name__ == "__main__":
    main()