"""
data_loader.py
──────────────
Load train/test/submission files theo config.
"""
import pandas as pd
from pathlib import Path


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, sample_submission.
    Returns: (train, test, sub)
    """
    paths = cfg["paths"]
    train = pd.read_csv(paths["train"])
    test  = pd.read_csv(paths["test"])
    sub   = pd.read_csv(paths["sub"])

    print(f"[DataLoader] Train : {train.shape}")
    print(f"[DataLoader] Test  : {test.shape}")
    print(f"[DataLoader] Sub   : {sub.shape}")
    return train, test, sub
