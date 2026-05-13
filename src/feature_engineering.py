import pandas as pd
import numpy as np

# ── Từng FE config là một hàm độc lập ─────────────────────────
def fe_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Không thêm gì — dùng làm baseline tham chiếu."""
    return df.copy()

def fe_v1_water(df: pd.DataFrame) -> pd.DataFrame:
    """Nhóm biến nước: Water_Deficit, Supply_Index, Residual."""
    d = df.copy()
    d["Water_Deficit"]       = d["Rainfall_mm"] - d["Soil_Moisture"]
    d["Water_Supply_Index"]  = d["Rainfall_mm"] + d["Soil_Moisture"]
    d["Irrigation_Residual"] = d["Previous_Irrigation_mm"] - d["Rainfall_mm"]
    return d

def fe_v2_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Nhóm khí hậu: Heat_Stress, Evapotransp, Humidity_Deficit."""
    d = df.copy()
    d["Heat_Stress_Index"] = d["Temperature_C"] * (1 - d["Humidity"] / 100)
    d["Humidity_Deficit"]  = 100 - d["Humidity"]
    d["Evapotransp_Proxy"] = (
        d["Temperature_C"] * d["Sunlight_Hours"] * (1 - d["Humidity"] / 100)
    )
    return d

def fe_v3_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Tương tác: Rainfall×Moisture, Temp×Humidity (scatter tách nhãn rõ)."""
    d = df.copy()
    d["Rainfall_x_Moisture"] = d["Rainfall_mm"] * d["Soil_Moisture"]
    d["Temp_x_Humidity"]     = d["Temperature_C"] * d["Humidity"]
    d["Prev_x_Area"]         = d["Previous_Irrigation_mm"] * d["Field_Area_hectare"]
    return d

def fe_v4_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize pH, Moisture, Rainfall, Temperature thành nhóm rời rạc.
    Tree-based models đôi khi học ngưỡng tốt hơn qua bins rõ ràng.
    """
    d = df.copy()
    d["pH_Group"] = pd.cut(
        d["Soil_pH"],
        bins=[0, 6.0, 7.5, 14],
        labels=["acid", "neutral", "alkaline"],
    )
    d["Moisture_Class"] = pd.cut(
        d["Soil_Moisture"],
        bins=3,
        labels=["dry", "moist", "wet"],
    )
    d["Rainfall_Level"] = pd.cut(
        d["Rainfall_mm"],
        bins=3,
        labels=["low", "medium", "high"],
    )
    d["Temp_Band"] = pd.cut(
        d["Temperature_C"],
        bins=3,
        labels=["cool", "warm", "hot"],
    )
    return d

def fe_full(df: pd.DataFrame) -> pd.DataFrame:
    """Tất cả features từ v1 + v2 + v3."""
    d = fe_v1_water(df)
    d = fe_v2_climate(d)
    d = fe_v3_interaction(d)
    return d

def fe_v5_full_plus(df: pd.DataFrame) -> pd.DataFrame:
    """v1 + v2 + v3 + v4 — toàn bộ features."""
    d = fe_full(df)
    d = fe_v4_bins(d)
    return d

# ── Registry: tên → hàm ───────────────────────────────────────
# Thêm config mới: định nghĩa hàm ở trên rồi đăng ký ở đây là xong.
# main.py sẽ tự động chạy config mới trong vòng experiment.
FE_REGISTRY = {
    "baseline"          : fe_baseline,
    "fe_v1_water"       : fe_v1_water,
    "fe_v2_climate"     : fe_v2_climate,
    "fe_v3_interaction" : fe_v3_interaction,
    "fe_v4_bins"        : fe_v4_bins,
    "fe_full"           : fe_full,
    "fe_v5_full_plus"   : fe_v5_full_plus,
}

def apply_fe(df: pd.DataFrame, config_name: str) -> pd.DataFrame:
    assert config_name in FE_REGISTRY, f"Unknown FE config: {config_name}"
    return FE_REGISTRY[config_name](df)