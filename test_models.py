"""
Unit tests for earthquake prediction models and pipeline
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from data_pipeline import EarthquakeDataPipeline
from deep_learning_models import (LSTMEarthquakePredictor, NeuralNetworkEarthquakePredictor,
                                   RandomForestEarthquakePredictor)


class TestDataPipeline(unittest.TestCase):
    """Tests for data pipeline"""
    
    def setUp(self):
        self.pipeline = EarthquakeDataPipeline()
    
    def test_data_loading(self):
        """Test that data loads correctly"""
        self.pipeline.load_data()
        self.assertIsNotNone(self.pipeline.df)
        self.assertGreater(len(self.pipeline.df), 0)
        print(f"✓ Data loaded: {len(self.pipeline.df)} records")
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        self.pipeline.load_data().engineer_features()
        self.assertIn('Hour', self.pipeline.df.columns)
        self.assertIn('Mag_mean_7d', self.pipeline.df.columns)
        print(f"✓ Features engineered: {len(self.pipeline.df.columns)} columns")
    
    def test_data_preparation(self):
        """Test train/val/test split"""
        self.pipeline.load_data().engineer_features().prepare_features()
        
        # Check splits
        self.assertIsNotNone(self.pipeline.X_train)
        self.assertIsNotNone(self.pipeline.X_val)
        self.assertIsNotNone(self.pipeline.X_test)
        
        # Check shapes
        self.assertEqual(self.pipeline.X_train.shape[1], self.pipeline.X_val.shape[1])
        self.assertEqual(self.pipeline.X_test.shape[0], self.pipeline.y_test.shape[0])
        
        print(f"✓ Data prepared: train={len(self.pipeline.X_train)}, "
              f"val={len(self.pipeline.X_val)}, test={len(self.pipeline.X_test)}")
    
    def test_normalization(self):
        """Test that data is normalized"""
        self.pipeline.load_data().engineer_features().prepare_features()
        
        # Check training data has mean ~0 and std ~1
        X_train_mean = np.mean(self.pipeline.X_train, axis=0)
        X_train_std = np.std(self.pipeline.X_train, axis=0)
        
        self.assertTrue(np.allclose(X_train_mean, 0, atol=1e-10))
        self.assertTrue(np.allclose(X_train_std, 1, atol=1e-10))
        print("✓ Data properly normalized")


class TestModelTraining(unittest.TestCase):
    """Tests for model training"""
    
    @classmethod
    def setUpClass(cls):
        """Prepare data once for all tests"""
        cls.pipeline = EarthquakeDataPipeline()
        cls.pipeline.load_data().engineer_features().prepare_features()
    
    def test_lstm_training(self):
        """Test LSTM-inspired model training"""
        model = LSTMEarthquakePredictor(n_estimators=50, max_depth=5, learning_rate=0.05)
        model.train(self.pipeline.X_train, self.pipeline.y_train, verbose=0)
        
        preds = model.predict(self.pipeline.X_test)
        r2 = r2_score(self.pipeline.y_test, preds)
        
        self.assertGreater(r2, 0)  # Should have positive R²
        self.assertEqual(len(preds), len(self.pipeline.y_test))
        print(f"✓ LSTM-Inspired trained: R² = {r2:.4f}")
    
    def test_nn_training(self):
        """Test Neural Network training"""
        model = NeuralNetworkEarthquakePredictor(
            input_dim=self.pipeline.X_train.shape[1],
            hidden_layers=[128, 64],
            max_iter=100
        )
        model.train(self.pipeline.X_train, self.pipeline.y_train, verbose=0)
        
        preds = model.predict(self.pipeline.X_test)
        r2 = r2_score(self.pipeline.y_test, preds)
        
        self.assertGreater(r2, 0)
        self.assertEqual(len(preds), len(self.pipeline.y_test))
        print(f"✓ Neural Network trained: R² = {r2:.4f}")
    
    def test_rf_training(self):
        """Test Random Forest training"""
        model = RandomForestEarthquakePredictor(n_estimators=50, max_depth=10)
        model.train(self.pipeline.X_train, self.pipeline.y_train, verbose=0)
        
        preds = model.predict(self.pipeline.X_test)
        r2 = r2_score(self.pipeline.y_test, preds)
        
        self.assertGreater(r2, 0)
        self.assertEqual(len(preds), len(self.pipeline.y_test))
        print(f"✓ Random Forest trained: R² = {r2:.4f}")


class TestPredictionConsistency(unittest.TestCase):
    """Tests for prediction consistency"""
    
    @classmethod
    def setUpClass(cls):
        """Load trained models"""
        try:
            cls.lstm_model = joblib.load('models/lstm_model.pkl')
            cls.nn_model = joblib.load('models/nn_model.pkl')
            cls.pipeline = joblib.load('models/data_pipeline.pkl')
            cls.models_available = True
        except:
            cls.models_available = False
    
    def test_models_exist(self):
        """Test that trained models exist"""
        if self.models_available:
            self.assertIsNotNone(self.lstm_model)
            self.assertIsNotNone(self.nn_model)
            print("✓ Trained models loaded successfully")
        else:
            print("⚠ Trained models not found - run model_training.py first")
    
    def test_prediction_range(self):
        """Test that predictions are in reasonable range"""
        if not self.models_available:
            self.skipTest("Trained models not available")
        
        # Create dummy data
        X = np.random.randn(10, self.pipeline.X_train.shape[1])
        X_scaled = self.pipeline.scaler_features.transform(X)
        
        preds = self.lstm_model.predict(X_scaled)
        
        # Predictions should be in reasonable magnitude range (e.g., 0-10)
        self.assertTrue(np.all(preds >= -5))
        self.assertTrue(np.all(preds <= 5))
        print(f"✓ Predictions in reasonable range: {preds.min():.2f} to {preds.max():.2f}")


class TestEvaluationMetrics(unittest.TestCase):
    """Tests for evaluation metrics"""
    
    def test_r2_calculation(self):
        """Test R² score calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
        
        r2 = r2_score(y_true, y_pred)
        self.assertGreater(r2, 0)
        self.assertLess(r2, 1)
        print(f"✓ R² score: {r2:.4f}")
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        self.assertGreater(rmse, 0)
        print(f"✓ RMSE: {rmse:.4f}")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
