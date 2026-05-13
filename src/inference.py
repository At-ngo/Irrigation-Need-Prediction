"""
inference.py
────────────
Export submission file và ghi experiment log.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def make_submission(
    final_preds: np.ndarray,
    sub_template: pd.DataFrame,
    target_col: str,
    task: str,
    output_dir: str = "outputs/",
    filename: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Tạo submission CSV đúng format."""
    sub = sub_template.copy()

    if task == "multiclass" and np.array(final_preds).ndim > 1:
        n_cls = final_preds.shape[1]
        for i in range(n_cls):
            sub[f"class_{i}"] = final_preds[:, i]
    else:
        sub[target_col] = final_preds

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname    = filename or f"submission_{ts}.csv"
    filepath = Path(output_dir) / fname
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(filepath, index=False)

    print(f"[Inference] Submission saved → {filepath}")
    print(sub.head(3).to_string())
    return sub, filepath


def log_experiment(
    cfg:          dict,
    results:      dict,
    weights:      dict,
    blend_score:  float,
    feature_cols: list[str],
    new_features: list[str],
    sub_path:     Path,
    lb_score:     float | None = None,
    notes:        str = "",
) -> pd.DataFrame:
    """
    Ghi lại experiment vào outputs/experiment_log.csv.
    Điền lb_score sau khi submit lên Kaggle.
    """
    metric     = cfg["metric"]
    output_dir = Path(cfg["paths"]["output"])

    entry = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task":          cfg["competition"]["task"],
        "metric":        metric,
        "n_folds":       cfg["cv"]["n_folds"],
        "n_features":    len(feature_cols),
        "n_new_features":len(new_features),
        "new_features":  str(new_features),
        "models":        str(list(weights.keys())),
        "weights":       str(weights),
        f"blend_{metric}": round(blend_score, 5),
        "lb_score":      lb_score,   # ← fill sau khi submit
        "sub_file":      str(sub_path.name),
        "notes":         notes,
    }

    # Per-model scores
    for name, res in results.items():
        entry[f"{name}_{metric}"] = round(res["mean"], 5)

    log_path = output_dir / "experiment_log.csv"
    log_df   = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    log_df   = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
    log_df.to_csv(log_path, index=False)

    print(f"\n[Inference] Experiment logged → {log_path}")
    print(json.dumps({k: v for k, v in entry.items()
                       if k not in ("weights", "new_features", "notes")}, indent=2))
    return log_df
