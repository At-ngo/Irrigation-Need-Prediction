"""
model.py
--------
Định nghĩa models + hàm run_cv dùng chung.
Hỗ trợ: LightGBM, XGBoost, CatBoost, Random Forest, Logistic Regression, SVM.
"""
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    balanced_accuracy_score,
)


# -- Metric --------------------------------------------------------------------
def compute_metric(y_true, y_pred, metric: str, task: str) -> float:
    """Tất cả metrics đều higher = better (loss metrics dùng negative)."""
    if metric == "auc":
        if task == "multiclass":
            return roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
        return roc_auc_score(y_true, y_pred)
    elif metric == "logloss":
        return -log_loss(y_true, y_pred)
    elif metric == "rmse":
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mae":
        return -mean_absolute_error(y_true, y_pred)
    elif metric == "rmsle":
        return -np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric in ("f1", "f1_macro"):
        avg = "macro" if metric == "f1_macro" else "binary"
        pred_label = (np.argmax(y_pred, axis=1) if np.array(y_pred).ndim > 1
                      else (np.array(y_pred) > 0.5).astype(int))
        return f1_score(y_true, pred_label, average=avg)
    elif metric == "accuracy":
        pred_label = (np.argmax(y_pred, axis=1) if np.array(y_pred).ndim > 1
                      else (np.array(y_pred) > 0.5).astype(int))
        return accuracy_score(y_true, pred_label)
    elif metric == "balanced_accuracy":
        pred_label = (np.argmax(y_pred, axis=1) if np.array(y_pred).ndim > 1
                      else (np.array(y_pred) > 0.5).astype(int))
        return balanced_accuracy_score(y_true, pred_label)
    raise ValueError(f"Unknown metric: {metric}")


# -- Core CV runner ------------------------------------------------------------
def run_cv(
    model_fn,
    train: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    metric: str,
    task: str,
    label: str = "model",
) -> dict:
    """
    K-fold cross-validation. Trả về OOF predictions + per-fold models.
    model_fn: callable(X_tr, y_tr, X_val, y_val) -> fitted model
    """
    fold_ids = sorted(train["fold"].unique())
    n_classes = int(train[target_col].nunique()) if task != "regression" else 1

    if task == "multiclass" and n_classes > 2:
        oof_preds = np.zeros((len(train), n_classes))
    else:
        oof_preds = np.zeros(len(train))

    scores, models = [], []
    t0 = time.time()

    print(f"\n{'-'*55}\n  Training: {label}\n{'-'*55}")

    for fold in fold_ids:
        tr_idx  = train["fold"] != fold
        val_idx = train["fold"] == fold
        X_tr  = train.loc[tr_idx,  feature_cols]
        y_tr  = train.loc[tr_idx,  target_col]
        X_val = train.loc[val_idx, feature_cols]
        y_val = train.loc[val_idx, target_col]

        model = model_fn(X_tr, y_tr, X_val, y_val)

        # Predict
        if task == "regression" or not hasattr(model, "predict_proba"):
            preds = model.predict(X_val)
        else:
            p = model.predict_proba(X_val)
            preds = p[:, 1] if task == "binary" else p

        oof_preds[val_idx.values] = preds
        score = compute_metric(y_val, preds, metric, task)
        scores.append(score)
        models.append(model)
        print(f"  Fold {fold+1}/{len(fold_ids)}  {metric}: {score:.5f}")

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    elapsed    = time.time() - t0

    print(f"\n  {'='*50}")
    print(f"  {label:15s} {metric.upper()}: {mean_score:.5f} ± {std_score:.5f}   {elapsed:.1f}s")
    print(f"  {'='*50}\n")

    return {
        "oof":    oof_preds,
        "scores": scores,
        "mean":   mean_score,
        "std":    std_score,
        "models": models,
        "label":  label,
        "time":   elapsed,
    }


# -- LightGBM ------------------------------------------------------------------
def build_lgbm_fn(params: dict, task: str):
    def lgbm_fn(X_tr, y_tr, X_val, y_val):
        Model = lgb.LGBMClassifier if task != "regression" else lgb.LGBMRegressor
        m = Model(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        return m
    return lgbm_fn


def get_lgbm_params(task: str, n_classes: int = 2, seed: int = 42) -> dict:
    base = {
        "n_estimators": 1000, "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": seed, "n_jobs": -1, "verbose": -1,
    }
    if task == "binary":
        base.update({"objective": "binary", "metric": "auc"})
    elif task == "multiclass":
        base.update({"objective": "multiclass", "metric": "multi_logloss",
                     "num_class": n_classes})
    else:
        base.update({"objective": "regression", "metric": "rmse"})
        base.pop("class_weight", None)
    return base


# -- XGBoost -------------------------------------------------------------------
def build_xgb_fn(params: dict, task: str):
    def xgb_fn(X_tr, y_tr, X_val, y_val):
        Model = xgb.XGBClassifier if task != "regression" else xgb.XGBRegressor
        m = Model(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return m
    return xgb_fn


def get_xgb_params(task: str, n_classes: int = 2, seed: int = 42) -> dict:
    base = {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": seed, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
    }
    if task == "binary":
        base.update({"objective": "binary:logistic", "eval_metric": "auc"})
    elif task == "multiclass":
        base.update({"objective": "multi:softprob", "num_class": n_classes,
                     "eval_metric": "mlogloss"})
    else:
        base.update({"objective": "reg:squarederror"})
    return base


# -- CatBoost ------------------------------------------------------------------
def build_cat_fn(params: dict, task: str, cat_feature_indices: list[int]):
    def cat_fn(X_tr, y_tr, X_val, y_val):
        Model = CatBoostClassifier if task != "regression" else CatBoostRegressor
        m = Model(**params)
        m.fit(
            Pool(X_tr, y_tr,  cat_features=cat_feature_indices),
            eval_set=Pool(X_val, y_val, cat_features=cat_feature_indices),
        )
        return m
    return cat_fn


def get_cat_params(task: str, seed: int = 42) -> dict:
    return {
        "iterations": 1000, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3, "random_seed": seed, "verbose": False,
        "early_stopping_rounds": 50, "auto_class_weights": "Balanced",
        "eval_metric": ("AUC" if task == "binary"
                        else "MultiClass" if task == "multiclass" else "RMSE"),
    }


# -- Random Forest -------------------------------------------------------------
def build_rf_fn(params: dict, task: str):
    def rf_fn(X_tr, y_tr, X_val, y_val):
        Model = RandomForestClassifier if task != "regression" else RandomForestRegressor
        m = Model(**params)
        m.fit(X_tr, y_tr)
        return m
    return rf_fn


def get_rf_params(task: str, seed: int = 42) -> dict:
    base = {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    if task == "regression":
        base.pop("class_weight", None)
    return base


# -- Logistic Regression ------------------------------------------------------
def build_lr_fn(params: dict, task: str):
    """Logistic Regression with StandardScaler pipeline."""
    def lr_fn(X_tr, y_tr, X_val, y_val):
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**params)),
        ])
        m.fit(X_tr, y_tr)
        return m
    return lr_fn


def get_lr_params(task: str, seed: int = 42) -> dict:
    return {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "multi_class": "multinomial" if task == "multiclass" else "auto",
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }


# -- SVM ----------------------------------------------------------------------
def build_svm_fn(params: dict, task: str):
    """SVM with StandardScaler pipeline. Dùng probability=True cho predict_proba."""
    def svm_fn(X_tr, y_tr, X_val, y_val):
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(**params) if task != "regression" else SVR(**params)),
        ])
        m.fit(X_tr, y_tr)
        return m
    return svm_fn


def get_svm_params(task: str, seed: int = 42) -> dict:
    base = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": seed,
    }
    if task != "regression":
        base["probability"] = True
        base["class_weight"] = "balanced"
    return base


# -- MODEL REGISTRY ------------------------------------------------------------
# Mỗi entry: (get_params_fn, build_fn, label)
# build_fn nhận (params, task) hoặc (params, task, cat_feature_indices) cho CatBoost
MODEL_REGISTRY = {
    "lgbm":     {"get_params": get_lgbm_params, "build": build_lgbm_fn, "label": "LightGBM"},
    "xgb":      {"get_params": get_xgb_params,  "build": build_xgb_fn,  "label": "XGBoost"},
    "catboost": {"get_params": get_cat_params,   "build": build_cat_fn,  "label": "CatBoost"},
    "rf":       {"get_params": get_rf_params,    "build": build_rf_fn,   "label": "RandomForest"},
    "lr":       {"get_params": get_lr_params,    "build": build_lr_fn,   "label": "LogisticRegression"},
    "svm":      {"get_params": get_svm_params,   "build": build_svm_fn,  "label": "SVM"},
}


def get_all_model_fns(task: str, n_classes: int = 3, seed: int = 42,
                      cat_feature_indices: list[int] | None = None,
                      enabled_models: list[str] | None = None):
    """
    Trả về dict {label: model_fn} cho tất cả models được bật.
    enabled_models: list tên model (key trong MODEL_REGISTRY). None = tất cả.
    """
    if enabled_models is None:
        enabled_models = list(MODEL_REGISTRY.keys())

    result = {}
    for name in enabled_models:
        if name not in MODEL_REGISTRY:
            print(f"  [WARN] Unknown model: {name}, skipping.")
            continue

        reg = MODEL_REGISTRY[name]
        if name == "catboost":
            params = reg["get_params"](task, seed)
            fn = reg["build"](params, task, cat_feature_indices or [])
        elif name in ("lgbm",):
            params = reg["get_params"](task, n_classes, seed)
            fn = reg["build"](params, task)
        elif name in ("xgb",):
            params = reg["get_params"](task, n_classes, seed)
            fn = reg["build"](params, task)
        else:
            params = reg["get_params"](task, seed)
            fn = reg["build"](params, task)

        result[reg["label"]] = fn

    return result


# -- Predict test --------------------------------------------------------------
def predict_test(results: dict, test: pd.DataFrame, feature_cols: list[str], task: str) -> np.ndarray:
    """Average predictions từ tất cả fold models."""
    preds_list = []
    for model in results["models"]:
        if task == "regression" or not hasattr(model, "predict_proba"):
            p = model.predict(test[feature_cols])
        else:
            pa = model.predict_proba(test[feature_cols])
            p  = pa[:, 1] if task == "binary" else pa
        preds_list.append(p)
    return np.mean(preds_list, axis=0)
