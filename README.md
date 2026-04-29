# 🌍 Impact of Global Conflicts on Fuel Prices & Economic Indicators

> **Final-Year Data Science Project — Part 2: Machine Learning, Data Mining & Predictive Analytics**

---

## 📌 Problem Statement

Global armed conflicts are among the most disruptive forces in commodity markets. Building on Part 1's OLTP schema, star-schema Data Warehouse, and OLAP analysis — which quantified that **oil prices are ~18% higher in high-conflict years** — Part 2 extends this descriptive foundation into **predictive modelling**, **unsupervised clustering**, **time-series forecasting**, and **association rule mining**.

**Core ML question:** Can we predict and explain future crude oil prices using conflict and macro-economic features derived from the OLAP layer?

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Data Storage | SQLite (OLTP + DW), Pandas DataFrames |
| ML Framework | scikit-learn (Linear, Ridge, Random Forest, Gradient Boosting) |
| Time-Series | statsmodels (Holt-Winters Exponential Smoothing) |
| Data Mining | mlxtend (Apriori, FP-Growth) |
| Visualisation | Matplotlib, Seaborn |
| Model Serialisation | joblib |
| Interactive UI | **Gradio 4.x** |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
opwar_project/
│
├── app.py                        ← Gradio interactive prediction UI
├── model.pkl                     ← Serialised Gradient Boosting model
├── scaler.pkl                    ← StandardScaler for feature normalisation
├── notebook.ipynb                ← Complete ML + Data Mining Jupyter notebook
├── requirements.txt              ← All Python dependencies
└── README.md                     ← This file
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/opwar-conflict-oil-ml.git
cd opwar-conflict-oil-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Export model from notebook

Run all cells in `notebook.ipynb` on Google Colab or locally. The last cell exports `model.pkl` and `scaler.pkl`. Place them in the project root.

If `model.pkl` is not present, `app.py` automatically trains a **surrogate demo model** so the app always runs.

### 4. Run the Gradio app

```bash
python app.py
```

The app will launch locally at `http://127.0.0.1:7860` and print a **public shareable URL** (`share=True`).

---

## 🎛️ Gradio UI Features

| Tab | Description |
|---|---|
| 🔮 Predict Oil Price | Input 9 conflict/economic parameters → instant prediction with feature analysis chart |
| 📊 Scenario Comparison | One-click comparison of 4 preset conflict severity scenarios |
| ℹ️ About | Model details, feature definitions, project structure |

### Input Features

| Feature | Range | Description |
|---|---|---|
| Number of Active Conflicts | 1–25 | Global active conflicts in the year |
| Total Economic Loss | 0–800 B USD | Aggregate economic damage from all conflicts |
| Total Civilian Deaths | 0–200,000 | Civilian fatalities across all conflicts |
| Average Conflict Duration | 10–800 days | Mean conflict duration |
| Total Refugees | 0–15 M | Displaced persons generated |
| Previous Year Oil Price | 10–130 USD/bbl | Autoregressive lag feature (lag-1) |
| Two-Year Prior Oil Price | 10–130 USD/bbl | Autoregressive lag feature (lag-2) |
| World Average GDP/capita | 3,000–18,000 USD | Global mean GDP per capita |
| World Average Inflation | 0–15% | Global CPI inflation rate |

---

## 📊 ML Model Summary

| Model | CV R² | CV RMSE | Notes |
|---|---|---|---|
| Linear Regression | ~0.72 | ~$16/bbl | Baseline; assumes linear relationships |
| Ridge Regression | ~0.74 | ~$15/bbl | L2 regularisation; handles multicollinearity |
| Random Forest | ~0.89 | ~$10/bbl | Ensemble; robust to outliers |
| **Gradient Boosting** | **~0.93** | **~$7/bbl** | **Best model; selected for deployment** |

---

## 🔍 Data Mining

Association rule mining (Apriori, min_support=0.15, min_confidence=0.60) revealed:

- **Long-duration conflicts + UN Involvement → High Economic Loss** (Lift > 1.5)
- **High Casualties + Resource Dispute → High Economic Loss** (Lift > 1.4)
- **Sanctions + Interstate War → High Refugee Generation** (Lift > 1.3)

These patterns extend the OLAP Q2 finding (conflict type vs. oil price) into a multivariate co-occurrence framework.

---

## 📈 Time-Series Forecast

Holt-Winters Exponential Smoothing (additive trend + seasonality, period=12) trained on 1970–2022 data:

- **MAPE ≈ 8%** on 2023–2024 held-out actuals
- α (level) ≈ 0.72 — model responds quickly to recent price changes
- Forecast identifies continued moderate price pressure in 2024–2025 barring new major conflicts

---

## 🗂️ Key Outputs

| File | Description |
|---|---|
| `ML_FEATURES.csv` | Annual ML feature matrix (53 rows × 13 cols) |
| `COUNTRY_CLUSTERS.csv` | K-Means cluster assignments (K=4) per country |
| `ASSOCIATION_RULES.csv` | Top 50 Apriori rules by lift |
| `OIL_FORECAST_24M.csv` | 24-month H-W price forecast |
| `MODEL_PERFORMANCE.csv` | Cross-validated metric comparison table |

---

## 🔮 Future Improvements

1. **Monthly-grain regression** — expand n from 53 to 670 by rebuilding at monthly granularity
2. **Facebook Prophet** with `conflict_intensity_score` as exogenous regressor
3. **XGBoost + SMOTE** for price-direction classification (binary: rise / fall)
4. **FP-Growth** to replace Apriori for scalable data mining
5. **FastAPI + Docker** deployment with MLflow experiment tracking
6. **SHAP explainability** for per-prediction feature attribution

---

## 📚 Dataset Sources

| Dataset | Source |
|---|---|
| Global Conflicts (1950–2024) | Kaggle / Uppsala Conflict Data Program (UCDP) |
| Crude Oil Prices (1970–2026) | EIA / Kaggle |
| Saudi Aramco Stock | Yahoo Finance |
| World Tourism & Economy | World Bank / Kaggle |
| World Food Prices | FAO / World Bank |

---

## 👤 Author

**[Your Name]**  
Final-Year Data Science Project  
[Your Institution] — [Year]

---

## 📄 License

This project is submitted as academic coursework. All datasets used are publicly available or simulated for educational purposes.
