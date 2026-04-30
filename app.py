#!/usr/bin/env python3
"""
Gradio Interactive Dashboard — Impact of Global Conflicts on Fuel Prices
Standalone deployment for Hugging Face Spaces (or local Gradio server)

This script:
1. Loads raw CSV data
2. Trains ML models (Ridge, Random Forest, Gradient Boosting, Linear)
3. Performs K-means clustering
4. Fits Holt-Winters time-series forecasting
5. Launches an interactive Gradio dashboard with multiple inference tabs
"""

import warnings
warnings.filterwarnings("ignore")

import io
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/Spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import gradio as gr

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def load_and_preprocess_data():
    """Load CSVs and prepare features for ML models."""
    
    print("📂 Loading datasets...")
    
    # Determine data paths (support both local and Hugging Face Spaces)
    base_dirs = ['.', 'Dataset', '/tmp/inputs']
    
    def find_file(filename):
        for base_dir in base_dirs + [os.path.dirname(__file__)]:
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                return path
        # Try just filename as fallback
        return filename
    
    # Load main datasets
    conflicts = pd.read_csv(find_file('global_conflicts_dataset.csv'))
    fuel = pd.read_csv(find_file('fuel_prices_1970_2026.csv'))
    
    # Optional: Load supplementary data if available (for richer features)
    aramco_data = None
    tourism_data = None
    food_data = None
    
    try:
        aramco_data = pd.read_csv(find_file('aramco.csv'))
    except:
        pass
    
    try:
        tourism_data = pd.read_csv(find_file('world_tourism_economy_data.csv'))
    except:
        pass
    
    try:
        food_data = pd.read_csv(find_file('WLD_RTFP_country_2023-10-02.csv'))
    except:
        pass
    
    print(f"✅ Conflicts: {conflicts.shape}")
    print(f"✅ Fuel prices: {fuel.shape}")
    
    # ── Clean conflicts data ──
    conflicts_clean = conflicts.dropna(subset=['Economic_Loss_USD_Billions', 'Civilian_Deaths', 'Year']).copy()
    conflicts_clean = conflicts_clean[conflicts_clean['Year'] >= 1970].copy()  # Align with fuel data
    
    # ── Clean fuel data ──
    fuel['Date'] = pd.to_datetime(fuel['Date'], errors='coerce')
    fuel_clean = fuel.dropna(subset=['Crude_Oil_Price']).copy()
    fuel_clean['Year'] = fuel_clean['Date'].dt.year
    fuel_clean['Month'] = fuel_clean['Date'].dt.month
    
    # ── Aggregate conflicts to yearly level ──
    conflicts_yearly = conflicts_clean.groupby('Year').agg({
        'Economic_Loss_USD_Billions': 'sum',
        'Civilian_Deaths': 'sum',
        'Military_Deaths': lambda x: x.sum() if 'Military_Deaths' in conflicts.columns else 0,
        'Duration_Days': 'mean',
        'Refugees_Millions': 'sum',
        'Country_A': 'count'  # conflict count
    }).reset_index()
    conflicts_yearly.columns = ['Year', 'Economic_Loss', 'Civilian_Deaths', 'Military_Deaths', 'Duration_Days', 'Refugees', 'Conflict_Count']
    
    # ── Aggregate fuel to yearly level ──
    fuel_yearly = fuel_clean.groupby('Year')['Crude_Oil_Price'].mean().reset_index()
    fuel_yearly.columns = ['Year', 'Oil_Price']
    
    # ── Merge ──
    ml_df = conflicts_yearly.merge(fuel_yearly, on='Year', how='inner')
    
    # ── Add economic indicators if tourism data available ──
    if tourism_data is not None:
        tourism_yearly = tourism_data.groupby('year').agg({'gdp': 'mean', 'inflation': 'mean'}).reset_index()
        tourism_yearly.columns = ['Year', 'GDP_Mean', 'Inflation_Mean']
        ml_df = ml_df.merge(tourism_yearly, on='Year', how='left')
    
    # ── Fill missing values ──
    ml_df = ml_df.fillna(ml_df.mean(numeric_only=True))
    
    print(f"✅ Merged dataset: {ml_df.shape}")
    print(f"   Years: {ml_df['Year'].min()} — {ml_df['Year'].max()}")
    
    return ml_df, fuel_clean, conflicts_clean


def prepare_ml_features(ml_df):
    """Prepare features and target for ML models."""
    
    # Remove rows with NaN in target
    ml_df = ml_df.dropna(subset=['Oil_Price']).copy()
    
    # Define features (exclude Year and Oil_Price)
    feature_cols = [col for col in ml_df.columns if col not in ['Year', 'Oil_Price']]
    
    X = ml_df[feature_cols].copy()
    y = ml_df['Oil_Price'].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    return X_scaled, y, feature_cols, scaler


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_models(X_scaled, y):
    """Train multiple regression models."""
    
    print("\n🤖 Training models...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"  ✅ {name:25s} | R²={r2:.4f} | RMSE=${rmse:.2f}/bbl")
    
    return results, X_test, y_test


def train_clustering(X_scaled):
    """Train K-means clustering on features."""
    
    print("\n📊 Training K-means clustering...")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  ✅ K-means (k=3) | Inertia={kmeans.inertia_:.2f}")
    print(f"  ✅ PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    return kmeans, pca, X_pca, clusters


def train_time_series_forecast(fuel_clean):
    """Train Holt-Winters exponential smoothing on fuel prices."""
    
    print("\n📈 Training time-series forecast (Holt-Winters)...")
    
    # Aggregate to monthly level
    fuel_ts = fuel_clean.set_index('Date')['Crude_Oil_Price'].resample('MS').mean()
    fuel_ts = fuel_ts.dropna()
    
    # Fit Holt-Winters
    try:
        hw_model = ExponentialSmoothing(
            fuel_ts,
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        )
        hw_fitted = hw_model.fit(optimized=True)
        print(f"  ✅ Holt-Winters fitted | AIC={hw_fitted.aic:.2f}")
        return hw_fitted, fuel_ts
    except:
        print("  ⚠️  Holt-Winters failed (insufficient data). Skipping...")
        return None, fuel_ts


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: GRADIO UI HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def make_gauge(predicted, lo=10, hi=160):
    """Draw a color gauge for oil price prediction."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    pct = (predicted - lo) / (hi - lo)
    pct = max(0.0, min(1.0, pct))
    
    # Color gradient
    if pct < 0.33:
        color = '#2ecc71'  # Green (cheap)
    elif pct < 0.66:
        color = '#f39c12'  # Orange (medium)
    else:
        color = '#e74c3c'  # Red (expensive)
    
    ax.barh([0], [pct], color=color, height=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.33, 0.66, 1])
    ax.set_xticklabels(['$10', '$56', '$103', '$160'])
    ax.set_yticks([])
    ax.set_title(f'Predicted Oil Price: ${predicted:.2f}/barrel', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig_to_pil(fig)


def plot_model_comparison(results):
    """Plot R² and RMSE for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    names = list(results.keys())
    r2_scores = [results[n]['r2'] for n in names]
    rmse_scores = [results[n]['rmse'] for n in names]
    
    ax1.barh(names, r2_scores, color='#3498db')
    ax1.set_xlabel('R² Score')
    ax1.set_title('Model R² Comparison')
    ax1.set_xlim(0, 1)
    
    ax2.barh(names, rmse_scores, color='#e74c3c')
    ax2.set_xlabel('RMSE ($/barrel)')
    ax2.set_title('Model RMSE Comparison')
    
    plt.tight_layout()
    return fig_to_pil(fig)


def plot_clustering(X_pca, clusters, centroids_pca):
    """Plot 2D clustering visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Country-Year Conflict Clusters (K-means)')
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    return fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: GRADIO INTERFACE TABS & CALLBACKS
# ═══════════════════════════════════════════════════════════════════

def predict_oil_price(economic_loss, civilian_deaths, duration_days, refugees, conflict_count):
    """Predict oil price given conflict features."""
    
    # Prepare feature vector
    input_data = np.array([economic_loss, civilian_deaths, duration_days, refugees, conflict_count]).reshape(1, -1)
    
    # Standardize using the trained scaler
    input_scaled = scaler.transform(input_data)
    
    # Use best model for prediction
    best_name = max(results, key=lambda k: results[k]['r2'])
    predicted_price = results[best_name]['model'].predict(input_scaled)[0]
    predicted_price = max(10, min(160, predicted_price))  # Clamp to reasonable range
    
    gauge = make_gauge(predicted_price)
    
    summary = f"""
    **Best Model:** {best_name} (R²={results[best_name]['r2']:.4f})
    
    **Predicted Oil Price:** ${predicted_price:.2f}/barrel
    
    **Input Features:**
    - Economic Loss: ${economic_loss:.1f}B
    - Civilian Deaths: {int(civilian_deaths):,}
    - Conflict Duration: {int(duration_days)} days
    - Refugees: {refugees:.1f}M
    - Conflict Count: {int(conflict_count)}
    """
    
    model_comp = plot_model_comparison(results)
    
    return gauge, summary, model_comp


def show_clustering_plot():
    """Return the clustering visualization."""
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    return plot_clustering(X_pca, clusters, centroids_pca)


def show_model_comparison():
    """Return model comparison plot."""
    return plot_model_comparison(results)


def show_forecast():
    """Generate time-series forecast plot."""
    if hw_fitted is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Forecast model not available (insufficient data)', ha='center', va='center')
        return fig_to_pil(fig)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Historical
    ax.plot(fuel_ts.index, fuel_ts.values, label='Historical', color='#3498db', linewidth=2)
    
    # Forecast
    forecast = hw_fitted.get_forecast(steps=24)
    forecast_index = pd.date_range(fuel_ts.index[-1], periods=25, freq='MS')[1:]
    ax.plot(forecast_index, forecast.mean_ci.iloc[:, 0], label='Forecast', color='#e74c3c', linewidth=2)
    ax.fill_between(forecast_index, forecast.conf_int()[:, 0], forecast.conf_int()[:, 1], alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Crude Oil Price ($/barrel)')
    ax.set_title('24-Month Oil Price Forecast (Holt-Winters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🚀 Impact of Global Conflicts on Fuel Prices")
    print("     Gradio Interactive Dashboard (Hugging Face Spaces Ready)")
    print("="*70 + "\n")
    
    # Load and prepare data
    ml_df, fuel_clean, conflicts_clean = load_and_preprocess_data()
    X_scaled, y, feature_cols, scaler = prepare_ml_features(ml_df)
    
    # Train models
    results, X_test, y_test = train_models(X_scaled, y)
    
    # Train clustering
    kmeans, pca, X_pca, clusters = train_clustering(X_scaled)
    
    # Train forecasting
    hw_fitted, fuel_ts = train_time_series_forecast(fuel_clean)
    
    # Determine best model
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_r2 = results[best_name]['r2']
    best_rmse = results[best_name]['rmse']
    
    # ─────────── BUILD GRADIO INTERFACE ──────────
    with gr.Blocks(title="Conflict-Fuel Price Analytics", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div style="text-align: center;">
            <h1>🌍 Impact of Global Conflicts on Fuel Prices & Economics</h1>
            <p style="font-size: 14px; color: #666;">
                Interactive ML Dashboard — Predict oil prices, cluster conflict regions, forecast market trends
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # ── TAB 1: PREDICT ──
            with gr.Tab("🔮 Predict"):
                gr.HTML("<h2>Predict Oil Price from Conflict Features</h2>")
                
                with gr.Row():
                    economic_loss = gr.Slider(0, 100, value=10, label="Economic Loss ($ Billions)")
                    civilian_deaths = gr.Slider(0, 100000, value=5000, label="Civilian Deaths", step=100)
                
                with gr.Row():
                    duration_days = gr.Slider(1, 5000, value=500, label="Conflict Duration (days)", step=10)
                    refugees = gr.Slider(0, 10, value=1, label="Refugees (Millions)", step=0.1)
                
                conflict_count = gr.Slider(1, 50, value=5, label="Number of Concurrent Conflicts", step=1)
                
                predict_btn = gr.Button("🚀 Predict", scale=2)
                
                with gr.Row():
                    gauge_output = gr.Image(label="Price Gauge", scale=1)
                    summary_output = gr.Markdown(label="Summary", scale=1)
                
                model_comp_output = gr.Image(label="Model Comparison", scale=2)
                
                predict_btn.click(
                    fn=predict_oil_price,
                    inputs=[economic_loss, civilian_deaths, duration_days, refugees, conflict_count],
                    outputs=[gauge_output, summary_output, model_comp_output]
                )
            
            # ── TAB 2: CLUSTER ──
            with gr.Tab("📊 Cluster"):
                gr.HTML("<h2>Country-Year Conflict Clusters (K-means, k=3)</h2>")
                cluster_btn = gr.Button("📈 Show Clusters", scale=2)
                cluster_plot = gr.Image(label="Clustering Visualization")
                
                cluster_btn.click(fn=show_clustering_plot, outputs=cluster_plot)
            
            # ── TAB 3: FORECAST ──
            with gr.Tab("📉 Forecast"):
                gr.HTML("<h2>24-Month Oil Price Forecast (Holt-Winters)</h2>")
                forecast_btn = gr.Button("📊 Generate Forecast", scale=2)
                forecast_plot = gr.Image(label="Forecast Chart")
                
                forecast_btn.click(fn=show_forecast, outputs=forecast_plot)
            
            # ── TAB 4: COMPARE MODELS ──
            with gr.Tab("🤖 Models"):
                gr.HTML("<h2>Model Performance Comparison</h2>")
                comp_btn = gr.Button("📊 Compare Models", scale=2)
                comp_plot = gr.Image(label="Performance Metrics")
                
                comp_btn.click(fn=show_model_comparison, outputs=comp_plot)
            
            # ── TAB 5: INFO ──
            with gr.Tab("ℹ️ Info"):
                gr.HTML(f"""
                <h2>Project Overview</h2>
                
                <p><strong>Best Model:</strong> {best_name} — R² = {best_r2:.4f}, RMSE = ${best_rmse:.2f}/barrel</p>
                
                <h3>Datasets</h3>
                <ul>
                    <li><strong>Global Conflicts 1950–2024:</strong> {conflicts_clean.shape[0]} events</li>
                    <li><strong>Crude Oil Prices 1970–2026:</strong> {fuel_clean.shape[0]} monthly observations</li>
                    <li><strong>Analysis Period:</strong> {int(ml_df['Year'].min())} — {int(ml_df['Year'].max())}</li>
                </ul>
                
                <h3>Key Findings (OLAP)</h3>
                <ul>
                    <li>Oil prices are <strong>~18% higher</strong> in high-conflict years</li>
                    <li><strong>Interstate Wars</strong> correlate with highest average oil prices</li>
                    <li>The <strong>2000s decade</strong> had the highest average price ($68.7/barrel)</li>
                </ul>
                
                <h3>Features Used</h3>
                <p>{', '.join(feature_cols)}</p>
                
                <h3>Technologies</h3>
                <p>Python · Pandas · Scikit-learn · Statsmodels · Gradio · Matplotlib · Seaborn</p>
                
                <hr style="margin: 20px 0;">
                <p style="font-size: 12px; color: #888;">
                    Final Year Project — Data Warehouse & BI with ML Analytics<br/>
                    Deployed on Hugging Face Spaces
                </p>
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align:center; color:#888; font-size:12px; margin-top:16px; padding:12px;
                    border-top: 1px solid #ddd;">
            🌍 Global Conflicts &amp; Fuel Price Analytics &nbsp;|&nbsp;
            ML Interactive Dashboard &nbsp;|&nbsp;
            Built with Gradio · Scikit-learn · Statsmodels
        </div>
        """)
    
    # Launch
    print("\n" + "="*70)
    print("  ✅ Gradio UI Ready!")
    print("="*70)
    print("\n  📍 Local: http://127.0.0.1:7860")
    print("  🌐 Public (share=True): https://xxxxx.gradio.live")
    print("\n")
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
