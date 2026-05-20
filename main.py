"""
main.py
-------
Pipeline chính:
  1. Chạy baseline comparison 6 models (LightGBM, XGBoost, CatBoost, RF, LR, SVM)
  2. Chọn best model -> SHAP analysis
  3. (Optional) Train full + xuất submission
"""
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
from train import run_experiment, run_baseline_comparison

RESULTS_PATH = "results/fe_results.csv"
os.makedirs("results",  exist_ok=True)
os.makedirs("outputs",  exist_ok=True)

TARGET    = "Irrigation_Need"
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
ID_MAP    = {v: k for k, v in LABEL_MAP.items()}   # 0->Low, 1->Medium, 2->High


# -- 1. Baseline comparison: 6 models --------------------------
def run_model_comparison(train_raw, cfg, fe_config="fe_full"):
    """
    So sánh 6 models trên cùng FE config.
    Lưu kết quả vào results/baseline_comparison.csv
    """
    print("\n" + "#" * 70)
    print("  STEP 1: BASELINE MODEL COMPARISON")
    print("#" * 70)

    enabled_models = ["lgbm", "xgb", "catboost", "rf"]

    results_df, all_models, feature_cols, X, y = run_baseline_comparison(
        df_raw=train_raw,
        cfg=cfg,
        fe_config_name=fe_config,
        apply_fe_fn=apply_fe,
        enabled_models=enabled_models,
    )

    # Lưu kết quả
    save_path = "results/baseline_comparison.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\n  Results saved -> {save_path}")

    return results_df, all_models, feature_cols, X, y


# -- 2. SHAP analysis với best model --------------------------
def run_shap_with_best(all_models, feature_cols, X, y, cfg,
                       best_model_label=None, results_df=None,
                       fe_config=None):
    """
    Chọn best model (hoặc chỉ định) -> chạy SHAP analysis.
    Lưu plots vào results/shap/{model}_{fe_config}/ để tránh ghi đè.
    """
    from shap_analysis import run_shap_analysis

    TASK = cfg["competition"]["task"]
    n_classes = y.nunique()

    if best_model_label is None and results_df is not None:
        best_model_label = results_df.iloc[0]["model"]

    # Tạo save_dir riêng theo model + fe_config
    if fe_config:
        save_dir = f"results/shap/{best_model_label}_{fe_config}"
    else:
        save_dir = "results/shap"

    print("\n" + "#" * 70)
    print(f"  STEP 2: SHAP ANALYSIS -- Model: {best_model_label} | FE: {fe_config or 'default'}")
    print("#" * 70)

    # Lấy model fold 0 (hoặc model nào tốt nhất)
    fold_models = all_models[best_model_label]
    best_model = fold_models[0]   # dùng model từ fold đầu

    shap_result = run_shap_analysis(
        model=best_model,
        X=X,
        y=y,
        feature_cols=feature_cols,
        model_label=best_model_label,
        task=TASK,
        n_classes=n_classes,
        max_samples=2000,
        save_dir=save_dir,
    )

    print(f"\n  SHAP plots saved -> {save_dir}/")
    return shap_result


# -- 3. FE experiments (giữ nguyên logic cũ) ------------------
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
    print(f"\nResults saved -> {RESULTS_PATH}")
    print(results_df.to_string(index=False))
    return results_df


# -- 4. Train full data + predict test ------------------------
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

    # Encode categorical -- fit trên train, transform cả hai
    cat_cols = [c for c in train_fe.columns
                if train_fe[c].dtype in ["object", "category"]
                and c not in drop_train]

    train_enc = pd.get_dummies(train_fe.drop(columns=drop_train), columns=cat_cols)
    test_enc  = pd.get_dummies(test_fe.drop(columns=drop_test),   columns=cat_cols)

    # Align columns -- test có thể thiếu 1 số dummy cols
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    feature_cols = train_enc.columns.tolist()
    y = train_fe[TARGET].map(LABEL_MAP)

    # -- K-fold: lưu OOF + test predictions -------------------
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

    # test_preds là average của predict() -> làm tròn về class gần nhất
    final_labels = np.round(test_preds).astype(int).clip(0, 2)
    return final_labels, oof_score


# -- 5. Xuất submission ----------------------------------------
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

    print(f"\n[Submission] Saved -> {fpath}")
    print(f"[Submission] OOF F1-macro: {oof_score:.4f}")
    print(f"[Submission] Label distribution:")
    print(sub[TARGET].value_counts().to_string())
    print(sub.head(5).to_string(index=False))
    return fpath


# -- Logger: ghi console ra file ---------------------------------
class TeeLogger:
    """Ghi stdout + stderr ra ca console lan file log."""
    def __init__(self, log_path):
        self.terminal_out = sys.stdout
        self.terminal_err = sys.stderr
        self.log_file = open(log_path, "w", encoding="utf-8")
        # Cũng redirect stderr
        self._stderr_tee = _TeeStream(self.terminal_err, self.log_file)

    def write(self, message):
        self.terminal_out.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal_out.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal_out
        sys.stderr = self.terminal_err

    def activate(self):
        """Redirect cả stdout và stderr."""
        sys.stdout = self
        sys.stderr = self._stderr_tee


class _TeeStream:
    """Ghi stderr ra cả terminal lẫn file log."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


LOG_CSV = "results/experiment_log.csv"

def append_log_csv(row: dict):
    """Ghi 1 dong vao CSV log ngay lap tuc (append mode)."""
    df_new = pd.DataFrame([row])
    if os.path.exists(LOG_CSV):
        df_new.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_CSV, index=False)


# -- Main: full grid FE x Model ----------------------------------
def main():
    with open("configs/config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_raw, test_raw, _ = load_data(cfg)
    id_col = cfg["competition"].get("id_col", "id")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup text log
    log_path = f"results/run_log_{ts}.txt"
    logger = TeeLogger(log_path)
    logger.activate()
    print(f"[LOG] Console output -> {log_path}")
    print(f"[LOG] CSV log -> {LOG_CSV}")
    print(f"[LOG] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    fe_configs = list(FE_REGISTRY.keys())
    enabled_models = ["lgbm", "xgb", "catboost", "rf"]

    all_rows = []          # cho bang tong hop
    submission_paths = []  # (fe, model, cv, path)

    print("\n" + "#" * 70)
    print(f"  FULL GRID: {len(fe_configs)} FE configs x {len(enabled_models)} models")
    print(f"  = {len(fe_configs) * len(enabled_models)} experiments")
    print("#" * 70)

    for fe_idx, fe_config in enumerate(fe_configs):
        print(f"\n\n{'*'*70}")
        print(f"  FE CONFIG [{fe_idx+1}/{len(fe_configs)}]: {fe_config}")
        print(f"{'*'*70}")

        # -- Train 4 models voi FE config nay ----------------------
        results_df, all_models, feature_cols, X, y = run_baseline_comparison(
            df_raw=train_raw,
            cfg=cfg,
            fe_config_name=fe_config,
            apply_fe_fn=apply_fe,
            enabled_models=enabled_models,
        )

        # -- Predict test + submission ----------------------------
        test_fe = apply_fe(test_raw, fe_config)
        drop_test = [c for c in [id_col] if c in test_fe.columns]
        cat_cols = [c for c in test_fe.columns
                    if test_fe[c].dtype in ["object", "category"]
                    and c not in drop_test]
        test_enc = pd.get_dummies(test_fe.drop(columns=drop_test), columns=cat_cols)
        test_enc = test_enc.reindex(columns=feature_cols, fill_value=0)

        for _, row in results_df.iterrows():
            model_label = row["model"]
            cv_score = row["cv_mean"]
            fold_models = all_models[model_label]

            test_preds = np.zeros(len(test_enc))
            for model in fold_models:
                pred = np.array(model.predict(test_enc[feature_cols])).ravel()
                test_preds += pred / len(fold_models)

            final_labels = np.round(test_preds).astype(int).clip(0, 2)

            fname = f"submission_{model_label}_{fe_config}_{ts}.csv"
            fpath = os.path.join("outputs", fname)
            sub = pd.DataFrame({
                id_col: test_raw[id_col],
                TARGET: [ID_MAP[i] for i in final_labels],
            })
            sub.to_csv(fpath, index=False)
            submission_paths.append((fe_config, model_label, cv_score, fpath))

            # -- Ghi CSV log ngay lap tuc -------------------------
            log_row = {
                "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fe_config":      fe_config,
                "model":          model_label,
                "cv_mean":        row["cv_mean"],
                "cv_std":         row["cv_std"],
                "time_sec":       row["time_sec"],
                "n_features":     row["n_features"],
                "submission_file": fname,
            }
            append_log_csv(log_row)
            all_rows.append(log_row)

        print(f"\n  Submissions saved for FE={fe_config}")

    # ============================================================
    #  BANG TONG HOP
    # ============================================================
    full_df = pd.DataFrame(all_rows).sort_values(
        "cv_mean", ascending=False
    ).reset_index(drop=True)

    save_path = "results/full_grid_results.csv"
    full_df.to_csv(save_path, index=False)

    # Pivot table: FE x Model
    pivot = full_df.pivot_table(
        index="fe_config", columns="model", values="cv_mean"
    ).reindex(index=fe_configs)

    print("\n\n" + "#" * 70)
    print("  FULL GRID RESULTS (balanced_accuracy)")
    print("#" * 70)
    print(pivot.to_string(float_format="{:.5f}".format))
    print(f"\n  Full results -> {save_path}")

    # Top 5
    print(f"\n  TOP 5:")
    print(full_df[["fe_config","model","cv_mean","cv_std","time_sec","n_features"]].head(5).to_string(index=False))

    # Best combo
    best = full_df.iloc[0]
    print(f"\n  BEST: {best['model']} + {best['fe_config']}  CV={best['cv_mean']:.5f}")

    print(f"\n  Submissions: {len(submission_paths)} files in outputs/")

    print(f"\n[LOG] Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[LOG] CSV log: {LOG_CSV}")
    print(f"[LOG] Text log: {log_path}")

    print("\n  Chon model roi chay SHAP:")
    print("    python main.py --shap <model_name> <fe_config>")
    print(f"  Vi du: python main.py --shap {best['model']} {best['fe_config']}")

    logger.close()


# -- SHAP: chay rieng sau khi da chon model ----------------------
def main_shap(model_name: str, fe_config: str = "fe_full"):
    """Chay SHAP analysis voi model da chon."""
    with open("configs/config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/shap_log_{model_name}_{fe_config}_{ts}.txt"
    logger = TeeLogger(log_path)
    logger.activate()
    print(f"[LOG] SHAP console output -> {log_path}")
    print(f"[LOG] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[LOG] Model: {model_name} | FE: {fe_config}")

    try:
        train_raw, _, _ = load_data(cfg)

        # Train lai model da chon de lay fitted model
        print(f"\nTraining {model_name} de chay SHAP...")
        results_df, all_models, feature_cols, X, y = run_baseline_comparison(
            df_raw=train_raw,
            cfg=cfg,
            fe_config_name=fe_config,
            apply_fe_fn=apply_fe,
            enabled_models=[{
                "LightGBM": "lgbm", "XGBoost": "xgb", "CatBoost": "catboost",
                "RandomForest": "rf", "LogisticRegression": "lr", "SVM": "svm",
            }.get(model_name, model_name)],
        )

        shap_result = run_shap_with_best(
            all_models, feature_cols, X, y, cfg,
            best_model_label=model_name,
            fe_config=fe_config,
        )

        save_dir = f"results/shap/{model_name}_{fe_config}"
        print("\n" + "#" * 70)
        print("  SHAP ANALYSIS DONE!")
        print(f"  Plots luu tai: {save_dir}/")
        print(f"  Log luu tai:   {log_path}")
        print("#" * 70)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()   # in ra cả stdout (TeeLogger) lẫn file log
        raise
    finally:
        print(f"[LOG] Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--shap":
        model_name = sys.argv[2]
        fe_config = sys.argv[3] if len(sys.argv) >= 4 else "fe_full"
        main_shap(model_name, fe_config)
    else:
        main()