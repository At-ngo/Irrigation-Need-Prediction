# 🌱 Irrigation Need Prediction — Kaggle Pipeline

Dự đoán mức độ tưới tiêu (`Low / Medium / High`) dựa trên các yếu tố khí hậu, đất đai và lịch sử canh tác. Đây là bài toán **multiclass classification** được xây dựng với pipeline tái sử dụng cho nhiều cuộc thi Kaggle.

---

## 📁 Cấu trúc thư mục

```
project/
│
├── configs/
│   └── config.yaml              # Cấu hình toàn bộ pipeline (paths, CV, metric, models...)
│
├── data/
│   └── raw/
│       ├── train.csv            # Dữ liệu huấn luyện
│       ├── test.csv             # Dữ liệu kiểm tra
│       └── sample_submission.csv
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
│
├── outputs/
│   └── submission_<config>_<timestamp>.csv   # File submission xuất ra
│
├── results/
│   └── fe_results.csv           # Kết quả so sánh các Feature Engineering configs
│
├── src/
│   ├── data_loader.py           # Load train/test/submission từ config
│   ├── feature_engineering.py  # Các FE configs + FE_REGISTRY
│   ├── preprocess.py            # Làm sạch dữ liệu + tạo CV folds
│   ├── train.py                 # Chạy cross-validation cho một FE config
│   ├── model.py                 # Định nghĩa LightGBM/XGBoost/CatBoost + run_cv
│   ├── inference.py             # Tạo submission CSV + ghi experiment log
│   └── validate.py              # So sánh kết quả FE, chọn config tốt nhất
│
├── main.py                      # Entry point — chạy toàn bộ pipeline
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md
```

---

## ⚙️ Cấu hình (`configs/config.yaml`)

Chỉnh file `config.yaml` để tuỳ biến pipeline cho từng cuộc thi:

| Trường | Ý nghĩa |
|---|---|
| `competition.task` | `binary` / `multiclass` / `regression` |
| `competition.target_col` | Tên cột nhãn |
| `competition.id_col` | Tên cột ID (hoặc `null`) |
| `paths.*` | Đường dẫn tới data và output |
| `cv.n_folds` | Số fold cross-validation |
| `cv.strategy` | `auto` / `stratified` / `kfold` / `group` |
| `metric` | Metric đánh giá (`f1_macro`, `auc`, `rmse`...) |

---

## 🔄 Workflow

```
config.yaml
    │
    ▼
[data_loader.py]          Load train / test / sample_submission
    │
    ▼
[preprocess.py]           Làm sạch (drop constant/missing cols) + tạo CV folds
    │
    ▼
[feature_engineering.py]  Áp dụng từng FE config trong FE_REGISTRY
    │
    ▼
[train.py]                Cross-validation với LightGBM (per FE config)
    │
    ▼
[validate.py]             So sánh kết quả → chọn FE config tốt nhất
    │
    ▼
[main.py]                 Train full data + predict test (tất cả configs)
    │
    ▼
[inference.py]            Xuất submission CSV + ghi experiment log
    │
    ▼
outputs/submission_*.csv
```

### Chạy pipeline

```bash
python main.py
```

Pipeline sẽ tự động:
1. Chạy **tất cả FE configs** trong `FE_REGISTRY`, lưu kết quả vào `results/fe_results.csv`
2. Train lại trên **toàn bộ dữ liệu** với từng config
3. Xuất file submission tương ứng vào `outputs/`

---

## 🧪 Feature Engineering Configs

Các config FE được định nghĩa trong `feature_engineering.py` và đăng ký tự động vào `FE_REGISTRY`:

| Config | Mô tả |
|---|---|
| `baseline` | Không thêm feature mới |
| `fe_v1_water` | Water Deficit, Supply Index, Irrigation Residual |
| `fe_v2_climate` | Heat Stress Index, Humidity Deficit, Evapotranspiration Proxy |
| `fe_v3_interaction` | Tương tác Rainfall×Moisture, Temp×Humidity, Prev×Area |
| `fe_v4_bins` | Discretize pH, Moisture, Rainfall, Temperature thành nhóm |
| `fe_full` | v1 + v2 + v3 |
| `fe_v5_full_plus` | v1 + v2 + v3 + v4 (toàn bộ) |

**Thêm FE config mới:** định nghĩa hàm trong `feature_engineering.py` rồi đăng ký vào `FE_REGISTRY` — pipeline sẽ tự động chạy thử.

---

## 📊 Models

| Model | Thư viện | Bật/tắt qua config |
|---|---|---|
| LightGBM | `lightgbm` | `models.lgbm: true` |
| XGBoost | `xgboost` | `models.xgb: true` |
| CatBoost | `catboost` | `models.catboost: true` |

Tất cả model đều hỗ trợ **early stopping** và **K-fold cross-validation**.

---

## 📈 Theo dõi kết quả

- So sánh FE configs: `results/fe_results.csv`
- Log experiment đầy đủ: `outputs/experiment_log.csv`
- Xem nhanh kết quả:

```bash
python src/validate.py
```

---

## 🔁 Tái sử dụng cho cuộc thi khác

1. Thay data trong `data/raw/`
2. Cập nhật `configs/config.yaml` (task, target_col, metric...)
3. Thêm FE functions phù hợp vào `feature_engineering.py`
4. Chạy `python main.py`