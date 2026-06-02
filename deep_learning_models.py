import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMEarthquakePredictor:
    """LSTM-inspired model using Gradient Boosting with temporal features"""
    
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05):
        """
        Gradient Boosting model to capture temporal patterns
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Tree depth
            learning_rate: Boosting learning rate
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None, verbose=1):
        """Train the gradient boosting model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if verbose:
            print(f"Training Gradient Boosting (LSTM-inspired)...")
        
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class NeuralNetworkEarthquakePredictor:
    """Deep neural network for earthquake magnitude prediction"""
    
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], learning_rate=0.001, max_iter=1000):
        """
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate
            max_iter: Maximum training iterations
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            alpha=0.001,  # L2 regularization
            batch_size=32,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            max_iter=max_iter,
            random_state=42,
            verbose=0
        )
        self.scaler = StandardScaler()
        self.history = {'loss': [], 'val_loss': []}
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None, verbose=1):
        """Train the neural network"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if verbose:
            print(f"Training Neural Network ({self.hidden_layers})...")
        
        self.model.fit(X_train_scaled, y_train)
        self.history['loss'] = self.model.loss_curve_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestEarthquakePredictor:
    """Random Forest model for comparison"""
    
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None, verbose=1):
        """Train random forest"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if verbose:
            print(f"Training Random Forest...")
        
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class EnsemblePredictor:
    """Ensemble combining multiple models"""
    
    def __init__(self, lstm_model=None, nn_model=None, rf_model=None, xgb_model=None):
        self.lstm_model = lstm_model
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.weights = {'lstm': 0.3, 'nn': 0.3, 'rf': 0.2, 'xgb': 0.2}
    
    def predict(self, X_lstm, X_nn, X_rf=None, X_xgb=None):
        """
        Ensemble prediction using weighted average
        
        Args:
            X_lstm: Features for LSTM (gradient boosting)
            X_nn: Features for NN
            X_rf: Features for Random Forest
            X_xgb: Features for XGBoost
        """
        predictions = {}
        total_weight = 0
        
        if self.lstm_model is not None and X_lstm is not None:
            pred = self.lstm_model.predict(X_lstm) * self.weights['lstm']
            predictions['lstm'] = pred
            total_weight += self.weights['lstm']
        
        if self.nn_model is not None and X_nn is not None:
            pred = self.nn_model.predict(X_nn) * self.weights['nn']
            predictions['nn'] = pred
            total_weight += self.weights['nn']
        
        if self.rf_model is not None and X_rf is not None:
            pred = self.rf_model.predict(X_rf) * self.weights['rf']
            predictions['rf'] = pred
            total_weight += self.weights['rf']
        
        if self.xgb_model is not None and X_xgb is not None:
            pred = self.xgb_model.predict(X_xgb) * self.weights['xgb']
            predictions['xgb'] = pred
            total_weight += self.weights['xgb']
        
        # Normalize weighted predictions
        if total_weight > 0:
            ensemble_pred = np.sum([v for v in predictions.values()], axis=0) / total_weight
        else:
            ensemble_pred = np.zeros_like(X_lstm[:, 0] if X_lstm is not None else X_nn[:, 0])
        
        return {
            'ensemble': ensemble_pred,
            'lstm': predictions.get('lstm'),
            'nn': predictions.get('nn'),
            'rf': predictions.get('rf'),
            'xgb': predictions.get('xgb')
        }
    
    def set_weights(self, lstm_weight=0.3, nn_weight=0.3, rf_weight=0.2, xgb_weight=0.2):
        """Adjust ensemble weights"""
        total = lstm_weight + nn_weight + rf_weight + xgb_weight
        self.weights = {
            'lstm': lstm_weight / total,
            'nn': nn_weight / total,
            'rf': rf_weight / total,
            'xgb': xgb_weight / total
        }

