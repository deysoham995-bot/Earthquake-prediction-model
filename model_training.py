import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import EarthquakeDataPipeline
from deep_learning_models import (LSTMEarthquakePredictor, NeuralNetworkEarthquakePredictor, 
                                   RandomForestEarthquakePredictor, EnsemblePredictor)
from model_evaluation import ModelEvaluator


def train_all_models():
    """Complete training pipeline for all models"""
    
    print("\n" + "="*70)
    print("🌍 EARTHQUAKE PREDICTION - DEEP LEARNING ENHANCEMENT")
    print("="*70)
    
    # ============= PHASE 1: DATA PREPARATION =============
    print("\n📊 PHASE 1: Data Preparation")
    print("-" * 70)
    
    pipeline = EarthquakeDataPipeline()
    pipeline.load_data().engineer_features().prepare_features()
    
    # Get regular features for all models
    X_train = pipeline.X_train
    X_val = pipeline.X_val
    X_test = pipeline.X_test
    y_train = pipeline.y_train
    y_val = pipeline.y_val
    y_test = pipeline.y_test
    
    # ============= PHASE 2: TRAIN LSTM-INSPIRED (GRADIENT BOOSTING) =============
    print("\n🔄 PHASE 2: Training LSTM-Inspired Model (Gradient Boosting)")
    print("-" * 70)
    
    lstm_model = LSTMEarthquakePredictor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )
    
    print("Training Gradient Boosting (temporal pattern capture)...")
    lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        verbose=1
    )
    
    lstm_preds = lstm_model.predict(X_test)
    lstm_mse = mean_squared_error(y_test, lstm_preds)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_r2 = r2_score(y_test, lstm_preds)
    lstm_mae = mean_absolute_error(y_test, lstm_preds)
    
    print(f"✅ LSTM-Inspired (Gradient Boosting) Results:")
    print(f"   MSE:  {lstm_mse:.6f}")
    print(f"   RMSE: {lstm_rmse:.6f}")
    print(f"   MAE:  {lstm_mae:.6f}")
    print(f"   R²:   {lstm_r2:.6f}")
    
    # Save LSTM model
    joblib.dump(lstm_model, 'models/lstm_model.pkl')
    print("💾 LSTM model saved to models/lstm_model.pkl")
    
    # ============= PHASE 3: TRAIN NEURAL NETWORK =============
    print("\n🧠 PHASE 3: Training Neural Network Model")
    print("-" * 70)
    
    nn_model = NeuralNetworkEarthquakePredictor(
        input_dim=X_train.shape[1],
        hidden_layers=[256, 128, 64],
        learning_rate=0.001,
        max_iter=500
    )
    
    print("Training Neural Network...")
    nn_model.train(
        X_train, y_train,
        X_val, y_val,
        verbose=1
    )
    
    nn_preds = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test, nn_preds)
    nn_rmse = np.sqrt(nn_mse)
    nn_r2 = r2_score(y_test, nn_preds)
    nn_mae = mean_absolute_error(y_test, nn_preds)
    
    print(f"✅ Neural Network Results:")
    print(f"   MSE:  {nn_mse:.6f}")
    print(f"   RMSE: {nn_rmse:.6f}")
    print(f"   MAE:  {nn_mae:.6f}")
    print(f"   R²:   {nn_r2:.6f}")
    
    # Save NN model
    joblib.dump(nn_model, 'models/nn_model.pkl')
    print("💾 NN model saved to models/nn_model.pkl")
    
    # ============= PHASE 4: TRAIN RANDOM FOREST =============
    print("\n🌲 PHASE 4: Training Random Forest Model")
    print("-" * 70)
    
    rf_model = RandomForestEarthquakePredictor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5
    )
    
    print("Training Random Forest...")
    rf_model.train(X_train, y_train, verbose=1)
    
    rf_preds = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    
    print(f"✅ Random Forest Results:")
    print(f"   MSE:  {rf_mse:.6f}")
    print(f"   RMSE: {rf_rmse:.6f}")
    print(f"   MAE:  {rf_mae:.6f}")
    print(f"   R²:   {rf_r2:.6f}")
    
    # Save Random Forest model
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("💾 Random Forest model saved to models/rf_model.pkl")
    
    # ============= PHASE 5: TRAIN/LOAD XGBOOST =============
    print("\n⚡ PHASE 5: Training XGBoost Model")
    print("-" * 70)
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    
    print(f"✅ XGBoost Results:")
    print(f"   MSE:  {xgb_mse:.6f}")
    print(f"   RMSE: {xgb_rmse:.6f}")
    print(f"   MAE:  {xgb_mae:.6f}")
    print(f"   R²:   {xgb_r2:.6f}")
    
    # Save XGBoost model
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    print("💾 XGBoost model saved to models/xgboost_model.pkl")
    
    # ============= PHASE 6: ENSEMBLE =============
    print("\n🎯 PHASE 6: Ensemble Prediction")
    print("-" * 70)
    
    ensemble = EnsemblePredictor(lstm_model, nn_model, rf_model, xgb_model)
    ensemble_preds_dict = ensemble.predict(X_test, X_test, X_test, X_test)
    ensemble_preds = ensemble_preds_dict['ensemble']
    
    ensemble_mse = mean_squared_error(y_test, ensemble_preds)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    
    print(f"✅ Ensemble Results:")
    print(f"   MSE:  {ensemble_mse:.6f}")
    print(f"   RMSE: {ensemble_rmse:.6f}")
    print(f"   MAE:  {ensemble_mae:.6f}")
    print(f"   R²:   {ensemble_r2:.6f}")
    
    # ============= PHASE 7: COMPARISON & SUMMARY =============
    print("\n" + "="*70)
    print("📈 MODEL COMPARISON SUMMARY")
    print("="*70)
    
    results = pd.DataFrame({
        'Model': ['LSTM-Inspired', 'Neural Network', 'Random Forest', 'XGBoost', 'Ensemble'],
        'MSE': [lstm_mse, nn_mse, rf_mse, xgb_mse, ensemble_mse],
        'RMSE': [lstm_rmse, nn_rmse, rf_rmse, xgb_rmse, ensemble_rmse],
        'MAE': [lstm_mae, nn_mae, rf_mae, xgb_mae, ensemble_mae],
        'R² Score': [lstm_r2, nn_r2, rf_r2, xgb_r2, ensemble_r2]
    })
    
    print("\n" + results.to_string(index=False))
    
    # Find best model
    best_idx = results['R² Score'].idxmax()
    best_model = results.loc[best_idx, 'Model']
    best_r2 = results.loc[best_idx, 'R² Score']
    
    print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.6f})")
    
    # Save results
    results.to_csv('models/model_comparison.csv', index=False)
    print("💾 Results saved to models/model_comparison.csv")
    
    # Save pipeline for inference
    joblib.dump(pipeline, 'models/data_pipeline.pkl')
    print("💾 Data pipeline saved to models/data_pipeline.pkl")
    
    # ============= PHASE 8: COMPREHENSIVE EVALUATION =============
    print("\n📊 PHASE 8: Comprehensive Evaluation & Visualization")
    print("-" * 70)
    
    # Inverse transform predictions and targets for evaluation
    y_test_original = pipeline.scaler_target.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    lstm_preds_original = pipeline.scaler_target.inverse_transform(lstm_preds.reshape(-1, 1)).ravel()
    nn_preds_original = pipeline.scaler_target.inverse_transform(nn_preds.reshape(-1, 1)).ravel()
    rf_preds_original = pipeline.scaler_target.inverse_transform(rf_preds.reshape(-1, 1)).ravel()
    xgb_preds_original = pipeline.scaler_target.inverse_transform(xgb_preds.reshape(-1, 1)).ravel()
    ensemble_preds_original = pipeline.scaler_target.inverse_transform(ensemble_preds.reshape(-1, 1)).ravel()
    
    # Create evaluator
    evaluator = ModelEvaluator(y_test_original)
    evaluator.add_prediction('LSTM-Inspired', lstm_preds_original)
    evaluator.add_prediction('Neural Network', nn_preds_original)
    evaluator.add_prediction('Random Forest', rf_preds_original)
    evaluator.add_prediction('XGBoost', xgb_preds_original)
    
    # Evaluate
    detailed_metrics = evaluator.evaluate_all()
    print("\n" + detailed_metrics.to_string())
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    evaluator.plot_actual_vs_predicted('models/evaluation/')
    evaluator.plot_residuals('models/evaluation/')
    evaluator.plot_error_distribution('models/evaluation/')
    evaluator.plot_metrics_comparison('models/evaluation/')
    evaluator.generate_report('models/evaluation/')
    
    print("\n" + "="*70)
    print("✅ TRAINING & EVALUATION COMPLETE!")
    print("="*70 + "\n")
    
    return {
        'lstm': lstm_model,
        'nn': nn_model,
        'rf': rf_model,
        'xgb': xgb_model,
        'ensemble': ensemble,
        'pipeline': pipeline,
        'results': results,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train all models
    models_dict = train_all_models()
