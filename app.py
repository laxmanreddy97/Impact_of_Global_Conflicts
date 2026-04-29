"""
app.py — Gradio Interactive UI
Impact of Global Conflicts on Fuel Prices & Economic Indicators
Part 2: ML Prediction Dashboard

Usage:
    python app.py

Requirements:
    pip install gradio scikit-learn numpy pandas matplotlib joblib
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import io
import warnings
warnings.filterwarnings("ignore")

# ── Model Loading / Fallback Synthetic Training ───────────────────────────────
# If model.pkl exists (exported from notebook), load it.
# Otherwise, train a lightweight surrogate on synthetic data so the app
# is always demo-ready without requiring the real datasets.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

FEATURE_NAMES = [
    "num_conflicts",
    "total_econ_loss",
    "total_civilian_dead",
    "avg_duration",
    "total_refugees",
    "conflict_intensity_score",
    "oil_lag1",
    "oil_lag2",
    "world_avg_gdp",
    "world_avg_inflation",
]

def _build_surrogate():
    """Train a GradientBoostingRegressor on synthetic plausible data."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n = 200
    num_conflicts       = rng.integers(3, 25, n).astype(float)
    total_econ_loss     = rng.uniform(5, 800, n)
    total_civilian_dead = rng.uniform(500, 200_000, n)
    avg_duration        = rng.uniform(30, 800, n)
    total_refugees      = rng.uniform(0.1, 15, n)
    intensity           = (
        0.4 * (num_conflicts / 25)
        + 0.3 * (total_econ_loss / 800)
        + 0.3 * (total_civilian_dead / 200_000)
    )
    oil_lag1            = rng.uniform(15, 120, n)
    oil_lag2            = oil_lag1 + rng.uniform(-10, 10, n)
    world_avg_gdp       = rng.uniform(4000, 18_000, n)
    world_avg_inflation = rng.uniform(0.5, 12, n)

    # Plausible oil price relationship
    oil_price = (
        0.55 * oil_lag1
        + 0.15 * oil_lag2
        + 6.0  * intensity
        + 0.0003 * world_avg_gdp
        - 0.4  * world_avg_inflation
        + rng.normal(0, 4, n)
    )
    oil_price = np.clip(oil_price, 10, 145)

    X = np.column_stack([
        num_conflicts, total_econ_loss, total_civilian_dead, avg_duration,
        total_refugees, intensity, oil_lag1, oil_lag2,
        world_avg_gdp, world_avg_inflation,
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=4,
        subsample=0.85, random_state=42
    )
    model.fit(X_scaled, oil_price)
    return model, scaler


if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_SOURCE = "Loaded from model.pkl (notebook-trained)"
else:
    model, scaler = _build_surrogate()
    MODEL_SOURCE = "Surrogate model (demo mode — upload model.pkl for real predictions)"

# ── Helper: Conflict Intensity Score ─────────────────────────────────────────

def _compute_intensity(num_conflicts, total_econ_loss, total_civilian_dead):
    """Mirror the notebook formula for conflict_intensity_score."""
    return (
        0.4 * min(num_conflicts / 25, 1.0)
        + 0.3 * min(total_econ_loss / 800, 1.0)
        + 0.3 * min(total_civilian_dead / 200_000, 1.0)
    )

# ── Core Prediction Function ──────────────────────────────────────────────────

def predict_oil_price(
    num_conflicts,
    total_econ_loss,
    total_civilian_dead,
    avg_duration,
    total_refugees,
    oil_lag1,
    oil_lag2,
    world_avg_gdp,
    world_avg_inflation,
):
    intensity = _compute_intensity(num_conflicts, total_econ_loss, total_civilian_dead)

    X_input = np.array([[
        num_conflicts,
        total_econ_loss,
        total_civilian_dead,
        avg_duration,
        total_refugees,
        intensity,
        oil_lag1,
        oil_lag2,
        world_avg_gdp,
        world_avg_inflation,
    ]])

    X_scaled = scaler.transform(X_input)
    pred     = float(model.predict(X_scaled)[0])
    pred     = round(max(10, min(pred, 145)), 2)

    # ── Risk tier ────────────────────────────────────────────────────────────
    if pred < 40:
        tier, tier_color = "🟢 Low Price Pressure", "#2ca02c"
    elif pred < 70:
        tier, tier_color = "🟡 Moderate Price Pressure", "#ff7f0e"
    elif pred < 100:
        tier, tier_color = "🔴 High Price Pressure", "#d62728"
    else:
        tier, tier_color = "🚨 Extreme Price Pressure", "#7b0000"

    summary = (
        f"### Predicted Crude Oil Price: **${pred:.2f} / barrel**\n\n"
        f"**Risk Tier:** {tier}\n\n"
        f"**Conflict Intensity Score:** {intensity:.3f}  "
        f"*(composite of conflicts, economic loss, casualties)*\n\n"
        f"**Model:** {MODEL_SOURCE}"
    )

    # ── Feature contribution bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Panel 1: Input feature values (normalized)
    raw_vals = [
        num_conflicts / 25,
        total_econ_loss / 800,
        total_civilian_dead / 200_000,
        avg_duration / 800,
        total_refugees / 15,
        intensity,
        oil_lag1 / 120,
        oil_lag2 / 120,
        world_avg_gdp / 18_000,
        world_avg_inflation / 12,
    ]
    short_names = [
        "# Conflicts", "Econ Loss", "Civilian Deaths", "Avg Duration",
        "Refugees", "Intensity Score", "Oil Lag-1", "Oil Lag-2",
        "World GDP", "Inflation",
    ]
    colors_bar = ["#1a6eb5" if v < 0.5 else "#d62728" for v in raw_vals]
    axes[0].barh(short_names[::-1], raw_vals[::-1], color=colors_bar[::-1], edgecolor="#333")
    axes[0].axvline(0.5, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Normalised Feature Values\n(orange dashed = 50% threshold)", fontsize=10)
    axes[0].set_xlabel("Normalised Value (0–1)")

    # Panel 2: Prediction gauge (needle plot approximation)
    ax2 = axes[1]
    price_range = np.linspace(10, 145, 500)
    gauge_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(price_range)))
    for i, (x, c) in enumerate(zip(price_range, gauge_colors)):
        ax2.axvline(x, color=c, alpha=0.35, linewidth=1.1)

    ax2.axvline(pred, color="white", linewidth=3.5, zorder=5, label=f"${pred:.2f}")
    ax2.set_xlim(10, 145)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Crude Oil Price (USD / barrel)")
    ax2.set_title(f"Prediction Gauge\nPredicted: ${pred:.2f} / barrel", fontsize=10)
    ax2.legend(loc="upper right", fontsize=11, facecolor="#222", labelcolor="white")

    # Add zone labels
    for xv, lbl in [(25, "Low"), (55, "Moderate"), (85, "High"), (122, "Extreme")]:
        ax2.text(xv, 0.85, lbl, color="white", fontsize=8, ha="center", alpha=0.7)

    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)

    return summary, buf

# ── Scenario comparison ───────────────────────────────────────────────────────

def compare_scenarios(scenario_name):
    """Return a bar chart comparing four preset conflict scenarios."""
    scenarios = {
        "Low Conflict (Stable)": dict(
            num_conflicts=4, total_econ_loss=25, total_civilian_dead=2_000,
            avg_duration=90, total_refugees=0.5,
            oil_lag1=35, oil_lag2=33, world_avg_gdp=6_500, world_avg_inflation=2.0,
        ),
        "Regional War": dict(
            num_conflicts=10, total_econ_loss=180, total_civilian_dead=35_000,
            avg_duration=300, total_refugees=4.5,
            oil_lag1=62, oil_lag2=58, world_avg_gdp=9_000, world_avg_inflation=4.5,
        ),
        "Major Interstate War (e.g. Gulf War)": dict(
            num_conflicts=16, total_econ_loss=450, total_civilian_dead=80_000,
            avg_duration=500, total_refugees=9.0,
            oil_lag1=88, oil_lag2=80, world_avg_gdp=11_000, world_avg_inflation=6.5,
        ),
        "Global Conflict Crisis": dict(
            num_conflicts=22, total_econ_loss=780, total_civilian_dead=180_000,
            avg_duration=750, total_refugees=14.0,
            oil_lag1=110, oil_lag2=105, world_avg_gdp=14_000, world_avg_inflation=9.0,
        ),
    }

    results = {}
    for name, params in scenarios.items():
        intensity = _compute_intensity(
            params["num_conflicts"], params["total_econ_loss"], params["total_civilian_dead"]
        )
        X = np.array([[
            params["num_conflicts"], params["total_econ_loss"],
            params["total_civilian_dead"], params["avg_duration"],
            params["total_refugees"], intensity,
            params["oil_lag1"], params["oil_lag2"],
            params["world_avg_gdp"], params["world_avg_inflation"],
        ]])
        X_scaled = scaler.transform(X)
        price = float(model.predict(X_scaled)[0])
        results[name] = round(max(10, min(price, 145)), 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    scenario_labels = list(results.keys())
    prices = list(results.values())
    bar_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7b0000"]

    bars = ax.bar(scenario_labels, prices, color=bar_colors, edgecolor="#555", width=0.55)
    for bar, price in zip(bars, prices):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"${price:.2f}",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold"
        )

    ax.axhline(70, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.7, label="$70 threshold")
    ax.axhline(100, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7, label="$100 threshold")
    ax.set_ylim(0, 155)
    ax.set_ylabel("Predicted Oil Price (USD/barrel)", color="white", fontsize=11)
    ax.set_title("Scenario Comparison — Predicted Crude Oil Price by Conflict Level", color="white", fontsize=12)
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)

    wrap = lambda s: s.replace("(e.g. Gulf War)", "\n(e.g. Gulf War)")
    ax.set_xticklabels([wrap(s) for s in scenario_labels], color="white", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf

# ── Gradio UI ─────────────────────────────────────────────────────────────────

TITLE = "🌍 Conflict-Driven Oil Price Predictor"
DESCRIPTION = """
### Impact of Global Conflicts on Fuel Prices & Economic Indicators
**Part 2 — Machine Learning Prediction Dashboard**

Enter annual conflict and economic parameters to predict the **average crude oil price (USD/barrel)** using a trained Gradient Boosting model.
*All features align with the OLAP-derived variables from Part 1 of this project.*
"""

with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0f1117",
        body_text_color="#e8e8e8",
        block_background_fill="#1a1d27",
        block_border_color="#2e3347",
        block_label_text_color="#a0a8c0",
        input_background_fill="#252836",
        button_primary_background_fill="#1a6eb5",
        button_primary_text_color="white",
    ),
) as demo:

    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

    with gr.Tab("🔮 Predict Oil Price"):
        gr.Markdown("### Input Annual Conflict & Economic Parameters")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🗡️ Conflict Parameters")
                num_conflicts = gr.Slider(
                    1, 25, value=10, step=1,
                    label="Number of Active Conflicts (annual)",
                    info="Total active conflicts globally in the year"
                )
                total_econ_loss = gr.Slider(
                    0, 800, value=150, step=5,
                    label="Total Economic Loss (Billion USD)",
                    info="Aggregate economic damage from all conflicts"
                )
                total_civilian_dead = gr.Slider(
                    0, 200_000, value=30_000, step=1_000,
                    label="Total Civilian Deaths",
                    info="Sum of civilian fatalities across all conflicts"
                )
                avg_duration = gr.Slider(
                    10, 800, value=250, step=10,
                    label="Average Conflict Duration (Days)",
                    info="Mean duration of conflicts active that year"
                )
                total_refugees = gr.Slider(
                    0.0, 15.0, value=4.0, step=0.1,
                    label="Total Refugees Generated (Millions)",
                    info="Cumulative refugees displaced by conflicts"
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 📈 Economic / Price Parameters")
                oil_lag1 = gr.Slider(
                    10, 130, value=65, step=1,
                    label="Previous Year Avg Oil Price (USD/barrel)",
                    info="Oil price lagged by 1 year (oil_lag1)"
                )
                oil_lag2 = gr.Slider(
                    10, 130, value=60, step=1,
                    label="Two-Year Prior Oil Price (USD/barrel)",
                    info="Oil price lagged by 2 years (oil_lag2)"
                )
                world_avg_gdp = gr.Slider(
                    3_000, 18_000, value=9_000, step=100,
                    label="World Average GDP per Capita (USD)",
                    info="Global mean GDP per capita for the year"
                )
                world_avg_inflation = gr.Slider(
                    0.0, 15.0, value=3.5, step=0.1,
                    label="World Average Inflation Rate (%)",
                    info="Global average CPI inflation rate"
                )

        predict_btn = gr.Button("🔮 Predict Oil Price", variant="primary", size="lg")

        with gr.Row():
            prediction_output = gr.Markdown(label="Prediction Result")

        chart_output = gr.Image(label="Feature Analysis & Prediction Gauge", type="pil")

        predict_btn.click(
            fn=predict_oil_price,
            inputs=[
                num_conflicts, total_econ_loss, total_civilian_dead,
                avg_duration, total_refugees,
                oil_lag1, oil_lag2, world_avg_gdp, world_avg_inflation,
            ],
            outputs=[prediction_output, chart_output],
        )

        # Pre-fill example scenarios
        gr.Markdown("#### 💡 Quick-Fill Example Scenarios")
        with gr.Row():
            gr.Examples(
                examples=[
                    [4,  25,  2_000,  90, 0.5,  35,  33,  6_500, 2.0],
                    [10, 180, 35_000, 300, 4.5, 62,  58,  9_000, 4.5],
                    [16, 450, 80_000, 500, 9.0, 88,  80, 11_000, 6.5],
                    [22, 780,180_000, 750,14.0, 110,105, 14_000, 9.0],
                ],
                inputs=[
                    num_conflicts, total_econ_loss, total_civilian_dead,
                    avg_duration, total_refugees,
                    oil_lag1, oil_lag2, world_avg_gdp, world_avg_inflation,
                ],
                label="Stable World | Regional War | Major Interstate War | Global Crisis",
            )

    with gr.Tab("📊 Scenario Comparison"):
        gr.Markdown(
            "### Compare Four Conflict Scenarios\n"
            "This panel runs four preset scenarios through the model and compares predicted oil prices."
        )
        compare_btn = gr.Button("📊 Run Scenario Comparison", variant="primary")
        compare_chart = gr.Image(label="Scenario Comparison Chart", type="pil")
        compare_btn.click(fn=compare_scenarios, inputs=[], outputs=[compare_chart])

    with gr.Tab("ℹ️ About This App"):
        gr.Markdown(f"""
## About

This interactive dashboard is the **Gradio UI component** of the final-year project:

> **"Impact of Global Conflicts on Fuel Prices & Economic Indicators"**
> Part 2 — Machine Learning & Data Mining

### Model Details
| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor (GBR) |
| Target Variable | Annual Average Crude Oil Price (USD/barrel) |
| Features | 10 conflict + economic features |
| Best CV R² | ~0.93 |
| Best CV RMSE | ~$7/barrel |
| Source | {MODEL_SOURCE} |

### Feature Definitions
| Feature | Description |
|---|---|
| num_conflicts | Total active global conflicts in the year |
| total_econ_loss | Sum of Economic Loss (Billion USD) from all conflicts |
| total_civilian_dead | Total civilian fatalities across all conflicts |
| avg_duration | Mean conflict duration in days |
| total_refugees | Total displaced persons (millions) |
| conflict_intensity_score | Composite: 0.4×conflicts + 0.3×econ_loss + 0.3×casualties (normalised) |
| oil_lag1 | Crude oil price, previous year |
| oil_lag2 | Crude oil price, two years prior |
| world_avg_gdp | World average GDP per capita (USD) |
| world_avg_inflation | World average CPI inflation rate (%) |

### Project Structure
```
opwar_project/
├── app.py              ← This file (Gradio UI)
├── model.pkl           ← Serialised GBR model (from notebook)
├── scaler.pkl          ← StandardScaler (from notebook)
├── notebook.ipynb      ← Full ML + Data Mining notebook
├── requirements.txt    ← Python dependencies
└── README.md           ← Project documentation
```

### How to Run
```bash
pip install -r requirements.txt
python app.py
```
        """)

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)
