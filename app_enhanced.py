import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from streamlit_folium import st_folium
from predict import EarthquakePredictionEngine
from live_earthquakes import LiveEarthquakeFetcher

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SeismoAI - Deep Learning Enhanced",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODERN UI STYLING
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #1a202c 100%);
}

h1 {
    color: #00d9ff;
    text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
}

.metric-card {
    background: #1a1f2e;
    border-left: 4px solid #00d9ff;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
}

.model-comparison {
    background: #0f1419;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_prediction_engine():
    try:
        return EarthquakePredictionEngine()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None

# =====================================================
# SIDEBAR - MODEL INFO & CONTROLS
# =====================================================
with st.sidebar:
    st.markdown("## 🎯 Model Configuration")
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models to Compare:",
        options=['LSTM-Inspired', 'Neural Network', 'Random Forest', 'XGBoost'],
        default=['LSTM-Inspired', 'XGBoost'],
        help="Choose which models to display in predictions"
    )
    
    # Model info
    st.markdown("---")
    st.markdown("### 📊 Model Performance (Test Set)")
    
    performance_data = {
        'LSTM-Inspired': {'R²': 0.3578, 'RMSE': 0.3662, 'MAE': 0.2685},
        'Neural Network': {'R²': 0.3151, 'RMSE': 0.3782, 'MAE': 0.2722},
        'Random Forest': {'R²': 0.3308, 'RMSE': 0.3738, 'MAE': 0.2761},
        'XGBoost': {'R²': 0.3490, 'RMSE': 0.3687, 'MAE': 0.2686}
    }
    
    for model_name, metrics in performance_data.items():
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{metrics['R²']:.3f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.3f}")
            with col3:
                st.metric("MAE", f"{metrics['MAE']:.3f}")
            st.caption(f"**{model_name}**", help=model_name)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **SeismoAI** combines 4 advanced ML models with ensemble methods for 
    accurate earthquake magnitude prediction.
    
    - **Dataset**: 18,030 California earthquakes (1966-2019)
    - **Features**: 18 engineered features including temporal windows
    - **Ensemble Best R²**: 0.364
    """)

# =====================================================
# MAIN CONTENT
# =====================================================
st.title("🌍 SeismoAI - Earthquake Magnitude Predictor")
st.markdown("**Advanced Deep Learning & ML Ensemble for Seismic Analysis**")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "📊 Model Analysis", "📈 Performance", "🌍 Live Earthquakes"])

# =====================================================
# TAB 1: PREDICTIONS
# =====================================================
with tab1:
    st.markdown("## Make Earthquake Magnitude Predictions")
    
    # Load engine
    engine = load_prediction_engine()
    
    if engine:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Earthquake Parameters")
            
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=35.5,
                step=0.1,
                help="Earthquake epicenter latitude"
            )
            
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-120.5,
                step=0.1,
                help="Earthquake epicenter longitude"
            )
            
            depth = st.number_input(
                "Depth (km)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Depth of earthquake hypocenter"
            )
            
            nst = st.slider(
                "Number of Seismic Stations",
                min_value=1,
                max_value=100,
                value=15,
                help="Number of seismic stations that recorded the event"
            )
            
            gap = st.slider(
                "Azimuthal Gap (degrees)",
                min_value=0,
                max_value=360,
                value=120,
                help="Largest azimuthal gap between stations"
            )
            
            clo = st.number_input(
                "Closest Station Distance (km)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5,
                help="Distance to closest seismic station"
            )
            
            rms = st.number_input(
                "RMS Error",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Root-mean-squared error of solution"
            )
        
        with col2:
            st.markdown("### 🕐 Temporal Context")
            
            date = st.date_input("Date", value=datetime.now())
            hour = st.slider("Hour (UTC)", 0, 23, 12)
            month = date.month
            year = date.year
            day_of_year = date.timetuple().tm_yday
            
            st.markdown("---")
            
            # Make prediction
            if st.button("🎯 Predict Magnitude", use_container_width=True, type="primary"):
                with st.spinner("🔮 Analyzing seismic data..."):
                    pred = engine.predict_single(
                        latitude=latitude,
                        longitude=longitude,
                        depth=depth,
                        nst=nst,
                        gap=gap,
                        clo=clo,
                        rms=rms,
                        year=year,
                        month=month,
                        hour=hour
                    )
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                # Ensemble prediction (main result)
                ensemble_pred = pred['ensemble']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Predicted Magnitude",
                        f"{ensemble_pred:.2f}",
                        delta=f"±0.37",
                        delta_color="off",
                        help="Ensemble prediction with ±0.37 RMSE confidence interval"
                    )
                
                with col2:
                    st.metric(
                        "Confidence Level",
                        f"{pred['confidence']:.1%}",
                        help="Based on ensemble R² score"
                    )
                
                with col3:
                    # Richter scale interpretation
                    if ensemble_pred < 3.0:
                        richter_label = "Micro"
                        color = "blue"
                    elif ensemble_pred < 5.0:
                        richter_label = "Minor"
                        color = "green"
                    elif ensemble_pred < 6.0:
                        richter_label = "Moderate"
                        color = "orange"
                    else:
                        richter_label = "Major"
                        color = "red"
                    
                    st.metric("Richter Scale", richter_label)
                
                # Individual model predictions
                st.markdown("### 📊 Individual Model Predictions")
                
                model_preds = pd.DataFrame({
                    'Model': ['LSTM-Inspired', 'Neural Network', 'Random Forest', 'XGBoost', 'Ensemble'],
                    'Magnitude': [
                        pred['lstm_inspired'],
                        pred['neural_network'],
                        pred['random_forest'],
                        pred['xgboost'],
                        pred['ensemble']
                    ]
                })
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        model_preds,
                        x='Model',
                        y='Magnitude',
                        color='Model',
                        title="Model Predictions Comparison",
                        color_discrete_sequence=['#00d9ff', '#0088ff', '#00ff88', '#ffdd00', '#ff0088']
                    )
                    fig.add_hline(y=ensemble_pred, line_dash="dash", line_color="red", 
                                 annotation_text="Ensemble", annotation_position="right")
                    fig.update_layout(showlegend=False, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Predictions Table**")
                    st.dataframe(model_preds.style.highlight_max(subset=['Magnitude']), 
                                use_container_width=True)
                
                # Input parameters summary
                with st.expander("📝 Input Parameters Summary"):
                    summary = pd.DataFrame({
                        'Parameter': ['Latitude', 'Longitude', 'Depth (km)', 'Nst', 'Gap (°)', 
                                     'Distance (km)', 'RMS', 'Date', 'Hour (UTC)'],
                        'Value': [latitude, longitude, depth, nst, gap, clo, rms, date, hour]
                    })
                    st.table(summary)

# =====================================================
# TAB 2: MODEL ANALYSIS
# =====================================================
with tab2:
    st.markdown("## 🧠 Deep Learning Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### LSTM-Inspired (Gradient Boosting)")
        st.info("""
        **Architecture**: Gradient Boosting Regressor
        - 200 estimators
        - Max depth: 6
        - Learning rate: 0.05
        
        **Why LSTM-Inspired?**
        - Captures temporal sequences through iterative refinement
        - Natural pattern discovery across time windows
        - Robust to outliers
        
        **Performance**:
        - R²: 0.3578 (Best individual model)
        - RMSE: 0.3662
        - MAE: 0.2685
        """)
    
    with col2:
        st.markdown("### Neural Network")
        st.info("""
        **Architecture**: 256 → 128 → 64 → 1
        - ReLU activation
        - L2 regularization (α=0.001)
        - Batch normalization
        - Early stopping on validation loss
        
        **Training**:
        - Optimizer: Adam (lr=0.001)
        - Max iterations: 500
        - Validation split: 10%
        
        **Performance**:
        - R²: 0.3151
        - RMSE: 0.3782
        - MAE: 0.2722
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Random Forest")
        st.info("""
        **Configuration**:
        - 200 decision trees
        - Max depth: 20
        - Min samples split: 5
        - Parallel processing
        
        **Strengths**:
        - Captures non-linear relationships
        - Feature importance scoring
        - Robust to outliers
        
        **Performance**:
        - R²: 0.3308
        - RMSE: 0.3738
        - MAE: 0.2761
        """)
    
    with col2:
        st.markdown("### XGBoost")
        st.info("""
        **Configuration**:
        - 200 boosting rounds
        - Max depth: 6
        - Learning rate: 0.08
        - Subsample: 0.8
        
        **Strengths**:
        - Industry-standard gradient boosting
        - Fast and scalable
        - Handles imbalanced data well
        
        **Performance**:
        - R²: 0.3490
        - RMSE: 0.3687
        - MAE: 0.2686
        """)

# =====================================================
# TAB 3: PERFORMANCE
# =====================================================
with tab3:
    st.markdown("## 📈 Model Performance Analysis")
    
    # Load evaluation data
    try:
        eval_data = pd.read_csv('models/evaluation/detailed_metrics.csv', index_col=0)
        comparison_data = pd.read_csv('models/model_comparison.csv', index_col=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### R² Score Comparison")
            fig = px.bar(
                eval_data.reset_index(),
                x='index',
                y='R² Score',
                title="R² Score by Model",
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### RMSE vs MAE")
            fig = px.scatter(
                eval_data.reset_index(),
                x='RMSE',
                y='MAE',
                text='index',
                title="Error Metrics Comparison",
                size_max=60
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📊 Detailed Metrics")
        st.dataframe(eval_data.round(6), use_container_width=True)
        
        # Load and display evaluation plots
        import os
        eval_path = 'models/evaluation/'
        
        if os.path.exists(eval_path):
            st.markdown("### 📸 Evaluation Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(f'{eval_path}actual_vs_predicted.png'):
                    st.image(f'{eval_path}actual_vs_predicted.png', 
                            caption="Actual vs Predicted Magnitudes")
            
            with col2:
                if os.path.exists(f'{eval_path}residuals.png'):
                    st.image(f'{eval_path}residuals.png',
                            caption="Residuals Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(f'{eval_path}error_distribution.png'):
                    st.image(f'{eval_path}error_distribution.png',
                            caption="Absolute Error Distribution")
            
            with col2:
                if os.path.exists(f'{eval_path}metrics_comparison.png'):
                    st.image(f'{eval_path}metrics_comparison.png',
                            caption="Metrics Comparison")
    
    except Exception as e:
        st.warning(f"Could not load evaluation data: {e}")
        st.info("Run `python model_training.py` to generate evaluation reports")
        
        
# =====================================================
# TAB 4: LIVE EARTHQUAKES
# =====================================================
with tab4:
    st.markdown("## 🌍 Real-Time Global Earthquake Monitor")

    # Sidebar-like controls inside tab
    col1, col2, col3 = st.columns(3)

    with col1:
        time_period = st.selectbox(
            "⏱️ Time Period",
            ["hour", "day", "week", "month"],
            index=1
        )

    with col2:
        min_magnitude = st.slider(
            "📈 Minimum Magnitude",
            0.0,
            10.0,
            2.5,
            0.1
        )

    with col3:
        auto_refresh = st.checkbox("🔄 Auto Refresh")

    # Auto refresh
    if auto_refresh:
        st.rerun()

    # Fetch live data
    with st.spinner("🌐 Fetching live earthquake data from USGS..."):
        live_df = LiveEarthquakeFetcher.fetch_earthquakes(
            time_period=time_period,
            min_magnitude=min_magnitude
        )

    if live_df.empty:
        st.error("❌ No earthquake data available.")
    else:

        # Statistics
        stats = LiveEarthquakeFetcher.get_statistics(live_df)

        st.markdown("### 📊 Live Earthquake Statistics")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "🌍 Total",
                stats['total']
            )

        with s2:
            st.metric(
                "🔴 Major (7+)",
                stats['major']
            )

        with s3:
            st.metric(
                "⚡ Max Magnitude",
                f"{stats['max_magnitude']:.1f}"
            )

        with s4:
            st.metric(
                "📏 Avg Depth",
                f"{stats['avg_depth']:.1f} km"
            )

        st.markdown("---")

        # WORLD MAP
        st.markdown("## 🗺️ Live Earthquake World Map")

        earthquake_map = LiveEarthquakeFetcher.create_map(
            live_df,
            center_lat=20,
            center_lon=0,
            zoom=2
        )

        st_folium(
            earthquake_map,
            width=None,
            height=700
        )

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Magnitude Distribution")

            fig_mag = px.histogram(
                live_df,
                x="Magnitude",
                nbins=30,
                title="Earthquake Magnitude Distribution",
                color_discrete_sequence=['#00d9ff']
            )

            fig_mag.update_layout(
                plot_bgcolor='#111',
                paper_bgcolor='#111',
                font_color='white'
            )

            st.plotly_chart(fig_mag, use_container_width=True)

        with col2:
            st.markdown("### 🌊 Depth Analysis")

            fig_depth = px.scatter(
                live_df,
                x="Depth_km",
                y="Magnitude",
                color="Magnitude",
                size="Magnitude",
                hover_data=["Location"],
                title="Depth vs Magnitude"
            )

            fig_depth.update_layout(
                plot_bgcolor='#111',
                paper_bgcolor='#111',
                font_color='white'
            )

            st.plotly_chart(fig_depth, use_container_width=True)

        st.markdown("---")

        # Recent earthquakes
        st.markdown("## 📋 Latest Earthquakes")

        display_df = live_df[[
            'Magnitude',
            'Location',
            'Depth_km',
            'Time_UTC',
            'Status'
        ]]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )

        st.markdown("---")

        # Top regions
        st.markdown("## 🌎 Most Active Regions")

        top_regions = LiveEarthquakeFetcher.get_top_regions(live_df)

        st.dataframe(
            top_regions,
            use_container_width=True
        )

        st.markdown("---")

        # Strong earthquakes
        strong_eq = live_df[live_df['Magnitude'] >= 5.0]

        if not strong_eq.empty:

            st.markdown("## 🚨 Significant Earthquakes")

            for idx, row in strong_eq.head(10).iterrows():

                magnitude = row['Magnitude']

                if magnitude >= 7:
                    level = "🔴 MAJOR"
                elif magnitude >= 6:
                    level = "🟠 STRONG"
                else:
                    level = "🟡 MODERATE"

                st.warning(
                    f"{level} | "
                    f"M {magnitude:.1f} | "
                    f"{row['Location']} | "
                    f"Depth: {row['Depth_km']:.1f} km"
                )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🌍 <b>SeismoAI</b> - Deep Learning Enhanced Earthquake Prediction</p>
    <p>Powered by Gradient Boosting, Neural Networks, and Ensemble Methods</p>
    <p><small>Dataset: 18,030 California earthquakes (1966-2019) | Model: Production-Ready ✅</small></p>
</div>
""", unsafe_allow_html=True)
