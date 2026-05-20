"""
train.py
--------
Chạy baseline comparison nhiều models + SHAP analysis.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, classification_report,
    confusion_matrix,
)
import lightgbm as lgb


def run_experiment(df_raw, cfg: dict, fe_config_name: str, apply_fe_fn,
                   random_state: int = None):
    """Chạy 1 FE config với LightGBM (giữ backward-compat)."""
    TARGET     = cfg["competition"]["target_col"]
    ID_COL     = cfg["competition"]["id_col"]
    N_SPLITS   = cfg["cv"]["n_folds"]
    random_state = random_state or cfg["cv"]["seed"]
    label_map  = {"Low": 0, "Medium": 1, "High": 2}

    df = apply_fe_fn(df_raw, fe_config_name)

    # Bỏ cột id nếu có
    drop_cols = [c for c in [ID_COL] if c and c in df.columns]
    cat_cols  = [c for c in df.columns
                 if df[c].dtype in ["object", "category"] and c != TARGET]
    X = pd.get_dummies(
            df.drop(columns=[TARGET] + drop_cols),
            columns=cat_cols
        )
    y = df[TARGET].map(label_map)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=random_state)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        fold_scores.append(score)
        print(f"  [{fe_config_name}] fold {fold+1}: F1-macro={score:.4f}")

    return {
        "fe_config" : fe_config_name,
        "cv_mean"   : float(np.mean(fold_scores)),
        "cv_std"    : float(np.std(fold_scores)),
        "n_features": X.shape[1],
        "fold_scores": fold_scores,
    }


# -- Multi-model baseline comparison ------------------------------------------
def run_baseline_comparison(df_raw, cfg: dict, fe_config_name: str,
                            apply_fe_fn, enabled_models: list[str] = None,
                            random_state: int = None):
    """
    Chạy baseline comparison với nhiều models trên cùng 1 FE config.
    Trả về DataFrame so sánh các model + dict chứa fitted models.

    Parameters
    ----------
    df_raw : pd.DataFrame
    cfg : dict (config.yaml)
    fe_config_name : str (key trong FE_REGISTRY)
    apply_fe_fn : callable
    enabled_models : list[str] hoặc None (4 models mặc định)
    random_state : int
    """
    from model import (
        get_all_model_fns, compute_metric,
    )
    import time

    TARGET     = cfg["competition"]["target_col"]
    TASK       = cfg["competition"]["task"]
    ID_COL     = cfg["competition"]["id_col"]
    N_SPLITS   = cfg["cv"]["n_folds"]
    METRIC     = cfg["metric"]
    random_state = random_state or cfg["cv"]["seed"]
    label_map  = {"Low": 0, "Medium": 1, "High": 2}

    if enabled_models is None:
        enabled_models = ["lgbm", "xgb", "catboost", "rf"]

    # -- Prepare data -----------------------------------------
    df = apply_fe_fn(df_raw, fe_config_name)

    drop_cols = [c for c in [ID_COL] if c and c in df.columns]
    cat_cols  = [c for c in df.columns
                 if df[c].dtype in ["object", "category"] and c != TARGET]
    X = pd.get_dummies(
            df.drop(columns=[TARGET] + drop_cols),
            columns=cat_cols
        )
    y = df[TARGET].map(label_map)


    feature_cols = X.columns.tolist()
    n_classes = y.nunique()

    # Get model functions
    model_fns = get_all_model_fns(
        task=TASK,
        n_classes=n_classes,
        seed=random_state,
        cat_feature_indices=[],    # đã one-hot encode
        enabled_models=enabled_models,
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=random_state)
    folds = list(skf.split(X, y))

    # -- Train each model -------------------------------------
    all_results = []
    all_models  = {}

    for model_label, model_fn in model_fns.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_label}")
        print(f"{'='*60}")

        fold_scores = []
        fold_models = []
        t0 = time.time()

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = model_fn(X_tr, y_tr, X_val, y_val)

            # Predict -- dùng predict_proba nếu có
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_val)
                pred_label = np.argmax(p, axis=1)
            else:
                pred_label = model.predict(X_val)
                pred_label = np.round(pred_label).astype(int).clip(0, n_classes - 1)

            # Tính metric
            if METRIC == "balanced_accuracy":
                score = balanced_accuracy_score(y_val, pred_label)
            elif METRIC == "f1_macro":
                score = f1_score(y_val, pred_label, average="macro")
            elif METRIC == "accuracy":
                from sklearn.metrics import accuracy_score as acc_score
                score = acc_score(y_val, pred_label)
            else:
                score = f1_score(y_val, pred_label, average="macro")

            fold_scores.append(score)
            fold_models.append(model)
            print(f"  Fold {fold_idx+1}/{N_SPLITS}  {METRIC}: {score:.5f}")

        elapsed = time.time() - t0
        mean_s  = float(np.mean(fold_scores))
        std_s   = float(np.std(fold_scores))

        print(f"\n  --> {model_label:20s} {METRIC}: {mean_s:.5f} ± {std_s:.5f}   {elapsed:.1f}s\n")

        all_results.append({
            "model":       model_label,
            "cv_mean":     mean_s,
            "cv_std":      std_s,
            "time_sec":    round(elapsed, 1),
            "n_features":  len(feature_cols),
            "fold_scores": fold_scores,
        })
        all_models[model_label] = fold_models

    # -- Summary table -----------------------------------------
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "fold_scores"}
        for r in all_results
    ]).sort_values("cv_mean", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISON -- FE: {fe_config_name}  |  Metric: {METRIC}")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"{'='*70}\n")

    return results_df, all_models, feature_cols, X, y