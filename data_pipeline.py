import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class EarthquakeDataPipeline:
    """Enhanced data pipeline for earthquake prediction with temporal features"""
    
    def __init__(self, data_path='Dataset/Earthquake_Data.csv'):
        self.data_path = data_path
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load and validate earthquake data"""
        possible_paths = [
            self.data_path,
            f'Earthquake-prediction-using-Machine-learning-models-main/{self.data_path}'
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Could not find {self.data_path}")
        
        self.df = pd.read_csv(data_path, sep=r'\s+')
        self.df.columns = [col.strip() for col in self.df.columns]
        print(f"✓ Loaded {len(self.df)} earthquake records")
        return self
    
    def engineer_features(self, temporal_windows=[7, 30]):
        """Advanced feature engineering with temporal statistics"""
        # Parse datetime
        self.df['Date'] = pd.to_datetime(self.df['Date(YYYY/MM/DD)'], errors='coerce')
        self.df['Time'] = pd.to_datetime(self.df['Time'], errors='coerce')
        
        # Extract temporal features
        self.df['Hour'] = self.df['Time'].dt.hour
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Sort by date for temporal features
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Rolling statistics (7-day and 30-day windows)
        for window in temporal_windows:
            self.df[f'Mag_mean_{window}d'] = self.df['Mag'].rolling(window=window, min_periods=1).mean()
            self.df[f'Mag_std_{window}d'] = self.df['Mag'].rolling(window=window, min_periods=1).std().fillna(0)
            self.df[f'Event_count_{window}d'] = self.df['Mag'].rolling(window=window, min_periods=1).count()
        
        # Depth categories
        self.df['Depth_category'] = pd.cut(self.df['Depth'], bins=[0, 5, 15, 25, 100], 
                                           labels=[0, 1, 2, 3]).astype(float)
        
        print("✓ Feature engineering complete")
        return self
    
    def prepare_features(self, train_split=0.7, val_split=0.15, random_state=42):
        """Prepare and normalize features for modeling"""
        # Select features
        base_features = ['Latitude', 'Longitude', 'Depth', 'Nst', 'Gap', 'Clo', 'RMS']
        temporal_features = ['Hour', 'DayOfYear', 'Month', 'Quarter', 'Depth_category',
                            'Mag_mean_7d', 'Mag_std_7d', 'Event_count_7d',
                            'Mag_mean_30d', 'Mag_std_30d', 'Event_count_30d']
        
        self.feature_names = base_features + temporal_features
        
        # Remove rows with NaN
        self.df = self.df.dropna(subset=self.feature_names + ['Mag'])
        
        X = self.df[self.feature_names].values.astype(np.float32)
        y = self.df['Mag'].values.astype(np.float32)
        
        # Split: train -> val+test (from remaining 30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-train_split, random_state=random_state
        )
        
        # Further split temp into val and test
        val_ratio = val_split / (1 - train_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=random_state
        )
        
        # Normalize using training data statistics
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_val_scaled = self.scaler_features.transform(X_val)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).ravel()
        
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train_scaled
        self.y_val = y_val_scaled
        self.y_test = y_test_scaled
        
        print(f"✓ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"✓ Features: {len(self.feature_names)} features")
        print(f"✓ Target range: {y.min():.2f} - {y.max():.2f}")
        
        return self
    
    def create_sequences(self, lookback=30):
        """Create sequences for LSTM training (returns X, y as sequences)"""
        def _make_sequences(X, y, lookback):
            X_seq, y_seq = [], []
            for i in range(lookback, len(X)):
                X_seq.append(X[i-lookback:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = _make_sequences(self.X_train, self.y_train, lookback)
        X_val_seq, y_val_seq = _make_sequences(self.X_val, self.y_val, lookback)
        X_test_seq, y_test_seq = _make_sequences(self.X_test, self.y_test, lookback)
        
        print(f"✓ Sequences created (lookback={lookback})")
        print(f"  Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
        
        return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq
    
    def get_raw_data(self):
        """Return unscaled test data for evaluation"""
        return self.df[self.feature_names], self.df['Mag']
