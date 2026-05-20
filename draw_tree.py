"""
draw_tree.py
────────────
Vẽ sơ đồ cây DecisionTreeClassifier cho từng FE config.
Lưu ảnh vào results/tree/{fe_config}/

Usage:
    python draw_tree.py                           # Vẽ cả 3: baseline, fe_v1_water, fe_full
    python draw_tree.py baseline                  # Chỉ 1 FE config
    python draw_tree.py baseline fe_v1_water      # 2 FE configs
    python draw_tree.py --depth 5                 # Thay đổi max_depth (default=4)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from data_loader import load_data
from feature_engineering import apply_fe

# ── Config ────────────────────────────────────────────────────
TARGET    = "Irrigation_Need"
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
CLASS_NAMES = ["Low", "Medium", "High"]
TREE_COLORS = ["#3498db", "#e74c3c", "#2ecc71"]   # Blue, Red, Green


def draw_decision_tree(train_raw, cfg, fe_config, max_depth=4, save_dir=None):
    """Train DecisionTreeClassifier + vẽ sơ đồ cây."""

    if save_dir is None:
        save_dir = f"results/tree/{fe_config}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Decision Tree — FE: {fe_config} | max_depth={max_depth}")
    print(f"{'='*60}")

    # -- Prepare data --
    df = apply_fe(train_raw, fe_config)
    id_col = cfg["competition"].get("id_col", "id")

    drop_cols = [c for c in [id_col] if c and c in df.columns]
    cat_cols = [c for c in df.columns
                if df[c].dtype in ["object", "category"] and c != TARGET]

    X = pd.get_dummies(df.drop(columns=[TARGET] + drop_cols), columns=cat_cols)
    y = df[TARGET].map(LABEL_MAP)

    feature_names = X.columns.tolist()
    print(f"  Features: {len(feature_names)}")

    # -- Train DecisionTree --
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=50,
    )
    dt.fit(X, y)

    train_acc = balanced_accuracy_score(y, dt.predict(X))
    print(f"  Train balanced_accuracy: {train_acc:.4f}")

    # -- Quick CV score --
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        dt_cv = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=50,
        )
        dt_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        val_pred = dt_cv.predict(X.iloc[val_idx])
        cv_scores.append(balanced_accuracy_score(y.iloc[val_idx], val_pred))
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"  CV balanced_accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # ── 1. Full tree diagram (high-res) ──────────────────────
    print(f"\n  Plotting full tree diagram...")
    n_nodes = dt.tree_.node_count
    fig_width = max(20, min(60, n_nodes * 2))
    fig_height = max(10, max_depth * 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True,
        ax=ax,
    )
    ax.set_title(
        f"Decision Tree — {fe_config} (depth={max_depth}, "
        f"CV={cv_mean:.4f}±{cv_std:.4f})",
        fontsize=16, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    fpath = os.path.join(save_dir, f"tree_full_depth{max_depth}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Saved: {fpath}")

    # ── 2. Compact tree (depth=3) nếu depth > 3 ─────────────
    if max_depth > 3:
        print(f"  Plotting compact tree (depth=3)...")
        dt_compact = DecisionTreeClassifier(
            max_depth=3,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=50,
        )
        dt_compact.fit(X, y)

        fig, ax = plt.subplots(figsize=(24, 10), dpi=150)
        plot_tree(
            dt_compact,
            feature_names=feature_names,
            class_names=CLASS_NAMES,
            filled=True,
            rounded=True,
            fontsize=10,
            proportion=True,
            ax=ax,
        )
        ax.set_title(
            f"Decision Tree (compact) — {fe_config} (depth=3)",
            fontsize=16, fontweight="bold", pad=20,
        )
        plt.tight_layout()
        fpath = os.path.join(save_dir, "tree_compact_depth3.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> Saved: {fpath}")

    # ── 3. Feature importance bar chart ──────────────────────
    print(f"  Plotting feature importance...")
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": dt.feature_importances_,
    }).sort_values("importance", ascending=True)
    importances = importances[importances["importance"] > 0]   # chỉ giữ > 0

    fig, ax = plt.subplots(figsize=(10, max(6, len(importances) * 0.35)), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
    ax.barh(importances["feature"], importances["importance"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(
        f"Decision Tree Feature Importance — {fe_config}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fpath = os.path.join(save_dir, "tree_feature_importance.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Saved: {fpath}")

    # ── 4. Text rules ─────────────────────────────────────────
    rules = export_text(dt, feature_names=feature_names, class_names=CLASS_NAMES)
    fpath = os.path.join(save_dir, "tree_rules.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"Decision Tree Rules — {fe_config} | depth={max_depth}\n")
        f.write(f"CV balanced_accuracy: {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write(rules)
    print(f"    -> Saved: {fpath}")

    # ── 5. Feature importance CSV ─────────────────────────────
    imp_full = pd.DataFrame({
        "feature": feature_names,
        "importance": dt.feature_importances_,
    }).sort_values("importance", ascending=False)
    fpath = os.path.join(save_dir, "tree_feature_importance.csv")
    imp_full.to_csv(fpath, index=False)
    print(f"    -> Saved: {fpath}")

    print(f"\n  All outputs -> {save_dir}/")
    return dt, cv_mean, cv_std


def main():
    # Parse arguments
    args = sys.argv[1:]
    max_depth = 4

    # Check for --depth flag
    fe_configs = []
    i = 0
    while i < len(args):
        if args[i] == "--depth" and i + 1 < len(args):
            max_depth = int(args[i + 1])
            i += 2
        else:
            fe_configs.append(args[i])
            i += 1

    # Default: 3 configs
    if not fe_configs:
        fe_configs = ["baseline", "fe_v1_water", "fe_full"]

    # Load config + data
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_raw, _, _ = load_data(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[Decision Tree Visualization] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  FE configs: {fe_configs}")
    print(f"  max_depth:  {max_depth}")

    results = []
    for fe_config in fe_configs:
        dt, cv_mean, cv_std = draw_decision_tree(
            train_raw, cfg, fe_config, max_depth=max_depth
        )
        results.append({
            "fe_config": fe_config,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "n_features_used": int((dt.feature_importances_ > 0).sum()),
            "n_features_total": len(dt.feature_importances_),
        })

    # Summary
    print("\n" + "#" * 60)
    print("  SUMMARY")
    print("#" * 60)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print(f"\n  Outputs in: results/tree/")
    print("#" * 60)


if __name__ == "__main__":
    main()
