# 🌍 Impact of Global Conflicts on Fuel Prices — Hugging Face Space Edition

Interactive Gradio dashboard for predicting oil prices from conflict indicators, clustering country-conflict patterns, and forecasting market trends.

**Live at:** https://huggingface.co/spaces/[YOUR-USERNAME]/Impact-of-Global-Conflicts

---

## 🎯 What This Space Does

This Space provides an **interactive ML dashboard** with 5 tabs:

1. **🔮 Predict** — Estimate crude oil prices based on conflict features (economic loss, casualties, duration, refugee displacement, conflict count)
2. **📊 Cluster** — Visualize K-means clustering of country-conflict profiles (2D PCA projection)
3. **📉 Forecast** — 24-month oil price forecast using Holt-Winters exponential smoothing
4. **🤖 Models** — Compare Multiple regression models (Linear, Ridge, Random Forest, Gradient Boosting) with R² and RMSE metrics
5. **ℹ️ Info** — Project overview, datasets, OLAP findings, and key insights

---

## 📊 Data Sources

The Space automatically loads and processes:

- **global_conflicts_dataset.csv** — 3,000+ conflict events (1950–2024)
  - Fields: Country, Year, Conflict Type, Economic Loss, Deaths, Duration, Refugees
- **fuel_prices_1970_2026.csv** — Monthly crude oil prices
  - Field: Monthly price, date range 1970–2026

Optional (if available):
- `aramco.csv` — Saudi Aramco stock data
- `world_tourism_economy_data.csv` — GDP & inflation per country-year
- `WLD_RTFP_country_2023-10-02.csv` — Food price index

---

## 🚀 How to Deploy to Hugging Face Spaces

### Option 1: Automatic (UI)

1. Go to **[https://huggingface.co/spaces/create](https://huggingface.co/spaces/create)**
2. Choose **Gradio** as the SDK
3. Select **Public** or **Private**
4. Name it (e.g., `Impact-of-Global-Conflicts`)
5. Copy-paste the contents of `app.py` into the Space's `app.py`
6. Ensure `requirements.txt` is in the repo
7. Ensure CSV files are also in the Space repo (upload under **Files** tab)
8. Space auto-builds and launches 🎉

### Option 2: Git Clone + Push

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/[YOUR-USERNAME]/Impact-of-Global-Conflicts
cd Impact-of-Global-Conflicts

# Copy files from this repo
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/*.csv .  # Copy all CSV files

# Push to Hugging Face
git add .
git commit -m "Initial deployment: Gradio dashboard for conflict-fuel analysis"
git push

# Your Space will auto-build and go live!
```

### Option 3: Hugging Face CLI

```bash
huggingface-cli repo create Impact-of-Global-Conflicts --type space --space-sdk gradio
cd Impact-of-Global-Conflicts
cp app.py requirements.txt *.csv .
git add .
git commit -m "Deploy Gradio app"
git push
```

---

## 📁 Files Required for Deployment

```
Impact-of-Global-Conflicts/
├── app.py                               ← Gradio app (main entry point)
├── requirements.txt                     ← Python dependencies
├── global_conflicts_dataset.csv         ← Conflict data
├── fuel_prices_1970_2026.csv           ← Oil price data
├── README.md                            ← This file
└── [Optional] *.csv                     ← Aramco, tourism, food data
```

**Note:** CSV files **must** be in the Space repo. Hugging Face Spaces can access local files but cannot download from external URLs in the startup.

---

## 💻 Run Locally

```bash
# Clone the repo (or download the files)
cd Impact_of_Global_Conflicts

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py

# Open http://localhost:7860 in your browser
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **UI** | Gradio 4.0+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn (Ridge, RF, GB), Statsmodels (Holt-Winters) |
| **Viz** | Matplotlib, Seaborn |
| **Deployment** | Hugging Face Spaces |

---

## 📈 Model Performance

The Space trains **4 regression models** on conflict-fuel price data:

| Model | R² Score | RMSE ($/bbl) |
|-------|----------|-------------|
| Linear Regression | ~0.52 | $12.5 |
| Ridge (α=1.0) | ~0.52 | $12.4 |
| **Random Forest** | **~0.68** | **$9.8** ← Best |
| Gradient Boosting | ~0.65 | $10.2 |

---

## 🔑 Key Findings (From OLAP Analysis)

1. **Oil prices are ~18% higher** in high-conflict years vs. low-conflict years
2. **Interstate Wars** correlate with the highest average oil prices
3. The **2000s decade** had the highest average oil price ($68.7/barrel)
4. **USA, Russia, Iraq** account for the highest total economic losses from conflicts

---

## 🐛 Troubleshooting

### CSV Files Not Found
- **Error:** `FileNotFoundError: global_conflicts_dataset.csv`
- **Solution:** Ensure all `.csv` files are uploaded to the Space repo (under **Files** tab)

### Slow Model Training
- **Note:** First load takes ~30–60 seconds (trains 4 models, clustering, forecasting)
- Models are fit in-memory, not cached. This is intentional for reproducibility.

### Out of Memory
- Hugging Face Spaces have **16GB RAM**
- If datasets are very large, add data preprocessing to reduce size (e.g., downsampling)

### Missing Dependencies
- **Error:** `ModuleNotFoundError: No module named 'X'`
- **Solution:** Add the package to `requirements.txt` and re-deploy

---

## 📝 Citation

If you use this Space in research, please cite:

```bibtex
@misc{conflict_fuel_prices_2026,
  title={Impact of Global Conflicts on Fuel Prices & Economics},
  author={[Your Name]},
  year={2026},
  note={Interactive ML Dashboard on Hugging Face Spaces},
  url={https://huggingface.co/spaces/[YOUR-USERNAME]/Impact-of-Global-Conflicts}
}
```

---

## 📧 Support

For questions or issues:
1. Check the **Info** tab in the dashboard
2. Open an issue on the original [GitHub repo](https://github.com/laxmanreddy97/Impact_of_Global_Conflicts)
3. Comment on the Space

---

## 📜 License

This project is licensed under the **MIT License** — see the original repo for details.

---

**Last updated:** April 2026  
**Deployed Platform:** Hugging Face Spaces  
**Status:** ✅ Active
