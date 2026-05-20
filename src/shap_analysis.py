"""
shap_analysis.py
----------------
SHAP analysis: chọn best model -> train trên toàn bộ data -> vẽ SHAP plots.
Lưu tất cả biểu đồ vào results/shap/.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import os
import sys
from pathlib import Path


SHAP_DIR = "results/shap"


def run_shap_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_label: str = "BestModel",
    task: str = "multiclass",
    n_classes: int = 3,
    max_samples: int = 2000,
    save_dir: str = SHAP_DIR,
):
    """
    Chạy SHAP trên model đã train.
    - Nếu model là tree-based (LightGBM, XGBoost, CatBoost, RF) -> dùng TreeExplainer
    - Nếu model là linear (LR) hoặc SVM -> dùng KernelExplainer (sampling)

    Vẽ & lưu:
    1. SHAP Summary (bee-swarm) plot
    2. SHAP Bar plot (mean |SHAP|)
    3. SHAP Heatmap
    4. Per-class SHAP nếu multiclass
    """
    os.makedirs(save_dir, exist_ok=True)
    X_data = X[feature_cols].copy()

    # -- Detect model type early (dùng cho sampling + explainer) --
    actual_model = model
    has_pipeline = False

    if hasattr(model, "named_steps"):
        has_pipeline = True
        scaler = model.named_steps.get("scaler")
        actual_model = model.named_steps.get("lr") or model.named_steps.get("svm")

    is_catboost = "catboost" in type(actual_model).__name__.lower()
    is_tree = any(name in type(actual_model).__name__.lower()
                  for name in ["lgbm", "xgb", "catboost", "randomforest",
                               "gradient", "lightgbm"])

    # -- CatBoost: dùng native SHAP (memory-efficient hơn shap.TreeExplainer)
    if is_catboost and max_samples > 500:
        print(f"  [SHAP] CatBoost detected: reducing max_samples {max_samples} -> 500 (OOM prevention)")
        max_samples = 500

    # -- Sampling cho SHAP (tránh OOM) -------------------------
    if len(X_data) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_data), max_samples, replace=False)
        X_shap = X_data.iloc[idx]
        y_shap = y.iloc[idx] if y is not None else None
        print(f"  [SHAP] Sampled {max_samples} / {len(X_data)} rows for SHAP.")
    else:
        X_shap = X_data
        y_shap = y

    # -- Transform nếu pipeline (LR, SVM) ----------------------
    if has_pipeline:
        scaler = model.named_steps.get("scaler")
        if scaler:
            X_shap_transformed = pd.DataFrame(
                scaler.transform(X_shap),
                columns=feature_cols,
                index=X_shap.index,
            )
        else:
            X_shap_transformed = X_shap
    else:
        X_shap_transformed = X_shap

    print(f"  [SHAP] Model type: {type(actual_model).__name__}")
    print(f"  [SHAP] Computing SHAP values ({len(X_shap_transformed)} samples)...")
    sys.stdout.flush()   # flush trước khi chạy SHAP (phòng crash)

    # -- Free memory trước khi chạy SHAP -----------------------
    import gc
    gc.collect()

    if is_catboost:
        # Dùng CatBoost native SHAP — nhanh hơn & ít RAM hơn shap.TreeExplainer
        from catboost import Pool
        print(f"  [SHAP] Using CatBoost native get_feature_importance(ShapValues)...")
        sys.stdout.flush()
        shap_pool = Pool(X_shap_transformed)
        raw_shap = actual_model.get_feature_importance(
            data=shap_pool, type="ShapValues"
        )
        print(f"  [SHAP] Raw SHAP shape: {raw_shap.shape}")

        n_feats = len(feature_cols)
        if raw_shap.ndim == 3:
            # CatBoost multiclass: (n_samples, n_classes, n_features+1)
            # Bỏ cột base value (cột cuối) → (n_samples, n_classes, n_features)
            shap_values = raw_shap[:, :, :n_feats]
            # Transpose → (n_samples, n_features, n_classes) cho tương thích
            shap_values = np.transpose(shap_values, (0, 2, 1))
        else:
            # Binary/regression: (n_samples, n_features+1)
            shap_values = raw_shap[:, :n_feats]

        explainer = None  # không cần explainer object
        print(f"  [SHAP] CatBoost native SHAP done! Shape: {shap_values.shape}")
    elif is_tree:
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_shap_transformed)
    else:
        # KernelExplainer -- dùng background nhỏ
        bg_size = min(100, len(X_shap_transformed))
        background = shap.sample(X_shap_transformed, bg_size)
        if hasattr(actual_model, "predict_proba"):
            explainer = shap.KernelExplainer(actual_model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(actual_model.predict, background)
        shap_values = explainer.shap_values(X_shap_transformed, nsamples=100)

    print(f"  [SHAP] SHAP values computed successfully!")
    sys.stdout.flush()

    # -- Chuẩn hoá shap_values --------------------------------
    # Multiclass: shap_values là list of arrays [n_samples x n_features] per class
    # hoặc 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # Danh sách per-class arrays
        shap_values_all = shap_values  # list of (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_values_all = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    else:
        # Binary hoặc regression -- single array
        shap_values_all = [shap_values]

    class_names = ["Low", "Medium", "High"] if n_classes == 3 else [f"Class_{i}" for i in range(n_classes)]

    # -- 1. Summary plot (bee-swarm) -- mean abs across classes --
    print("  [SHAP] Plotting Summary (bar) ...")
    fig, ax = plt.subplots(figsize=(12, 8))
    if len(shap_values_all) > 1:
        # Mean absolute SHAP across classes
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values_all], axis=0)
        mean_importance = np.mean(mean_abs_shap, axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": mean_importance,
        }).sort_values("importance", ascending=True)

        ax.barh(feature_importance["feature"], feature_importance["importance"],
                color="#4C78A8", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean |SHAP value| (avg across classes)", fontsize=12)
        ax.set_title(f"SHAP Feature Importance -- {model_label}", fontsize=14, fontweight="bold")
    else:
        mean_importance = np.mean(np.abs(shap_values_all[0]), axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": mean_importance,
        }).sort_values("importance", ascending=True)

        ax.barh(feature_importance["feature"], feature_importance["importance"],
                color="#4C78A8", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(f"SHAP Feature Importance -- {model_label}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(save_dir, "shap_importance_bar.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Saved: {fpath}")

    # -- 2. Bee-swarm per class --------------------------------
    for c_idx, sv in enumerate(shap_values_all):
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}"
        print(f"  [SHAP] Plotting bee-swarm for class: {c_name} ...")

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_shap_transformed, feature_names=feature_cols,
                          show=False, max_display=20)
        plt.title(f"SHAP Summary -- {model_label} -- {c_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(save_dir, f"shap_beeswarm_{c_name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> Saved: {fpath}")

    # -- 3. SHAP Bar plot (top 20) per class -------------------
    for c_idx, sv in enumerate(shap_values_all):
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}"
        print(f"  [SHAP] Plotting bar for class: {c_name} ...")

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_shap_transformed, feature_names=feature_cols,
                          plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Bar -- {model_label} -- {c_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(save_dir, f"shap_bar_{c_name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> Saved: {fpath}")

    # -- 4. Feature importance table --------------------------
    importance_df = feature_importance.sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(save_dir, "shap_feature_importance.csv"), index=False)
    print(f"\n  [SHAP] Feature importance saved -> {save_dir}/shap_feature_importance.csv")
    print(importance_df.head(20).to_string(index=False))

    return {
        "shap_values": shap_values_all,
        "feature_importance": importance_df,
        "X_shap": X_shap_transformed,
        "explainer": explainer,
    }
