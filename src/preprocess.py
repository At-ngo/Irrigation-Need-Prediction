"""
preprocess.py
─────────────
Làm sạch dữ liệu cơ bản + tạo CV folds.
Không chứa feature engineering — để riêng trong feature_engineering.py.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


def basic_clean(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    id_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop constant columns và columns missing > 90%.
    Giữ nguyên target và id.
    """
    protected = {c for c in [target_col, id_col] if c}
    drop_cols = []

    for col in train.columns:
        if col in protected:
            continue
        if train[col].nunique() <= 1:
            drop_cols.append(col)
            print(f"  [Clean] Drop constant col   : {col}")
        elif train[col].isnull().mean() > 0.90:
            pct = train[col].isnull().mean() * 100
            drop_cols.append(col)
            print(f"  [Clean] Drop high-missing col: {col} ({pct:.0f}%)")

    drop_cols = list(set(drop_cols))
    train = train.drop(columns=drop_cols)
    test  = test.drop(columns=[c for c in drop_cols if c in test.columns])
    print(f"[Clean] After cleaning — Train: {train.shape}  Test: {test.shape}")
    return train, test


def make_folds(
    df: pd.DataFrame,
    target_col: str,
    n_folds: int = 5,
    strategy: str = "auto",
    task: str = "binary",
    group_col: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo cột 'fold' cho cross-validation.
    strategy: 'auto' | 'stratified' | 'kfold' | 'group'
    """
    df = df.copy()

    if strategy == "auto":
        strategy = "stratified" if task in ("binary", "multiclass") else "kfold"

    if strategy == "stratified":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(df, df[target_col])
    elif strategy == "group" and group_col:
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(df, df[target_col], df[group_col])
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(df)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(split_iter):
        df.loc[val_idx, "fold"] = fold

    print(f"[Folds] Strategy: {strategy}  |  {n_folds} folds created")
    counts = df["fold"].value_counts().sort_index()
    for fold, cnt in counts.items():
        print(f"  Fold {fold}: {cnt:,} samples")
    return df
