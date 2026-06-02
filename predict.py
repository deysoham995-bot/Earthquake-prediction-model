"""
Inference script for earthquake magnitude prediction
Uses trained models to make predictions on new data
"""

import numpy as np
import pandas as pd
import joblib
from deep_learning_models import EnsemblePredictor


class EarthquakePredictionEngine:
    """Production-ready prediction engine"""
    
    def __init__(self):
        self.pipeline = joblib.load('models/data_pipeline.pkl')
        self.lstm_model = joblib.load('models/lstm_model.pkl')
        self.nn_model = joblib.load('models/nn_model.pkl')
        self.rf_model = joblib.load('models/rf_model.pkl')
        self.xgb_model = joblib.load('models/xgboost_model.pkl')
        self.ensemble = EnsemblePredictor(self.lstm_model, self.nn_model, self.rf_model, self.xgb_model)
    
    def predict_single(self, latitude, longitude, depth, nst, gap, clo, rms,
                      year=2024, month=6, hour=12):
        """
        Make prediction for a single earthquake event
        
        Args:
            latitude, longitude: Location coordinates
            depth: Depth in kilometers
            nst: Number of seismic stations
            gap: Azimuthal gap
            clo: Distance of closest station
            rms: Root-mean-squared residual
            year, month, hour: Temporal features
        
        Returns:
            dict with predictions from all models and ensemble
        """
        # Create feature array
        X = np.array([[latitude, longitude, depth, nst, gap, clo, rms,
                      hour, 100, month, 2, 0, 4.5, 0.5, 3, 4.5, 0.5, 3]])
        
        # Normalize
        X_scaled = self.pipeline.scaler_features.transform(X)
        
        # Make predictions
        lstm_pred = self.lstm_model.predict(X_scaled)[0]
        nn_pred = self.nn_model.predict(X_scaled)[0]
        rf_pred = self.rf_model.predict(X_scaled)[0]
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        
        # Ensemble
        ensemble_dict = self.ensemble.predict(X_scaled, X_scaled, X_scaled, X_scaled)
        ensemble_pred = ensemble_dict['ensemble'][0]
        
        # Inverse transform
        lstm_mag = self.pipeline.scaler_target.inverse_transform([[lstm_pred]])[0][0]
        nn_mag = self.pipeline.scaler_target.inverse_transform([[nn_pred]])[0][0]
        rf_mag = self.pipeline.scaler_target.inverse_transform([[rf_pred]])[0][0]
        xgb_mag = self.pipeline.scaler_target.inverse_transform([[xgb_pred]])[0][0]
        ensemble_mag = self.pipeline.scaler_target.inverse_transform([[ensemble_pred]])[0][0]
        
        return {
            'lstm_inspired': float(lstm_mag),
            'neural_network': float(nn_mag),
            'random_forest': float(rf_mag),
            'xgboost': float(xgb_mag),
            'ensemble': float(ensemble_mag),
            'confidence': 0.36  # Based on ensemble R² score
        }
    
    def predict_batch(self, df):
        """
        Make predictions for multiple events
        
        Args:
            df: DataFrame with columns [latitude, longitude, depth, nst, gap, clo, rms, ...]
        
        Returns:
            DataFrame with predictions
        """
        # Extract features
        feature_cols = ['Latitude', 'Longitude', 'Depth', 'Nst', 'Gap', 'Clo', 'RMS',
                       'Hour', 'DayOfYear', 'Month', 'Quarter', 'Depth_category',
                       'Mag_mean_7d', 'Mag_std_7d', 'Event_count_7d',
                       'Mag_mean_30d', 'Mag_std_30d', 'Event_count_30d']
        
        X = df[feature_cols].values.astype(np.float32)
        X_scaled = self.pipeline.scaler_features.transform(X)
        
        # Predictions
        lstm_preds = self.lstm_model.predict(X_scaled)
        nn_preds = self.nn_model.predict(X_scaled)
        rf_preds = self.rf_model.predict(X_scaled)
        xgb_preds = self.xgb_model.predict(X_scaled)
        
        # Inverse transform
        lstm_mags = self.pipeline.scaler_target.inverse_transform(lstm_preds.reshape(-1, 1)).ravel()
        nn_mags = self.pipeline.scaler_target.inverse_transform(nn_preds.reshape(-1, 1)).ravel()
        rf_mags = self.pipeline.scaler_target.inverse_transform(rf_preds.reshape(-1, 1)).ravel()
        xgb_mags = self.pipeline.scaler_target.inverse_transform(xgb_preds.reshape(-1, 1)).ravel()
        
        # Ensemble
        ensemble_dict = self.ensemble.predict(X_scaled, X_scaled, X_scaled, X_scaled)
        ensemble_mags = self.pipeline.scaler_target.inverse_transform(
            ensemble_dict['ensemble'].reshape(-1, 1)
        ).ravel()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'LSTM-Inspired': lstm_mags,
            'Neural Network': nn_mags,
            'Random Forest': rf_mags,
            'XGBoost': xgb_mags,
            'Ensemble': ensemble_mags,
            'Average': (lstm_mags + nn_mags + rf_mags + xgb_mags) / 4
        })
        
        return results


if __name__ == "__main__":
    # Example usage
    engine = EarthquakePredictionEngine()
    
    # Single prediction
    print("Single Earthquake Prediction Example:")
    print("-" * 50)
    pred = engine.predict_single(
        latitude=35.5,
        longitude=-120.5,
        depth=10.0,
        nst=15,
        gap=120,
        clo=5.0,
        rms=0.05
    )
    
    print(f"LSTM-Inspired:   {pred['lstm_inspired']:.2f}")
    print(f"Neural Network:  {pred['neural_network']:.2f}")
    print(f"Random Forest:   {pred['random_forest']:.2f}")
    print(f"XGBoost:         {pred['xgboost']:.2f}")
    print(f"Ensemble:        {pred['ensemble']:.2f} (Confidence: {pred['confidence']:.2%})")
