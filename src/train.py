import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

def run_experiment(df_raw, cfg: dict, fe_config_name: str, apply_fe_fn,
                   random_state: int = None):
    TARGET     = cfg["competition"]["target_col"]
    ID_COL     = cfg["competition"]["id_col"]
    N_SPLITS   = cfg["cv"]["n_folds"]
    random_state = random_state or cfg["cv"]["seed"]
    label_map  = {"Low": 0, "Medium": 1, "High": 2}

    df = apply_fe_fn(df_raw, fe_config_name)

    # Bỏ cột id nếu có
    drop_cols = [c for c in [ID_COL] if c and c in df.columns]
    cat_cols  = [c for c in df.columns
                 if df[c].dtype == "object" and c != TARGET]
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