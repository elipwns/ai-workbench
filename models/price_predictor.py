#!/usr/bin/env python3
"""
Price Prediction Models
Basic ML models for price forecasting based on sentiment + technical indicators
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.s3_manager import S3DataManager
from datetime import datetime

class PricePredictor:
    def __init__(self):
        self.s3_manager = S3DataManager()
        self.models = {}
        
    def load_ml_dataset(self, symbol='BTC'):
        """Load ML dataset from S3"""
        try:
            # Get most recent ML features file
            objects = self.s3_manager.s3_client.list_objects_v2(
                Bucket=self.s3_manager.bucket_name,
                Prefix=f"raw-data/ml_features_{symbol}_"
            )
            
            if not objects.get('Contents'):
                print(f"No ML dataset found for {symbol}")
                return pd.DataFrame()
            
            # Get most recent file
            latest = sorted(objects['Contents'], 
                          key=lambda x: x['LastModified'], 
                          reverse=True)[0]
            
            response = self.s3_manager.s3_client.get_object(
                Bucket=self.s3_manager.bucket_name,
                Key=latest['Key']
            )
            
            from io import StringIO
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            print(f"Loaded ML dataset: {len(df)} samples, {len(df.columns)} features")
            return df
            
        except Exception as e:
            print(f"Error loading ML dataset: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepare features for ML training"""
        if df.empty:
            return None, None, None, None
        
        # Select feature columns (exclude targets and metadata)
        feature_cols = [col for col in df.columns if not col.startswith('target_') 
                       and col not in ['timestamp', 'date', 'symbol']]
        
        # Remove non-numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(0)  # Fill NaN with 0
        
        # Targets
        y_direction = df['target_direction_1d'].fillna(0).astype(int)  # Classification
        y_return = df['target_return_1d'].fillna(0)  # Regression
        
        print(f"Features: {len(X.columns)}")
        print(f"Samples with targets: {len(y_direction[y_direction.notna()])}")
        
        return X, y_direction, y_return, numeric_cols.tolist()
    
    def train_direction_model(self, X, y):
        """Train Random Forest for price direction (up/down)"""
        if len(X) < 10:
            print("Not enough data for training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        print(f"Direction Model - Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 Features:")
        print(feature_importance.head())
        
        return rf
    
    def train_return_model(self, X, y):
        """Train Random Forest for price return prediction"""
        if len(X) < 10:
            print("Not enough data for training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_mae = mean_absolute_error(y_train, rf.predict(X_train))
        test_mae = mean_absolute_error(y_test, rf.predict(X_test))
        
        print(f"Return Model - Train MAE: {train_mae:.3f}%, Test MAE: {test_mae:.3f}%")
        
        return rf
    
    def train_models(self, symbol='BTC'):
        """Train both direction and return models"""
        print(f"Training models for {symbol}...")
        
        # Load data
        df = self.load_ml_dataset(symbol)
        if df.empty:
            print("No data available for training")
            return
        
        # Prepare features
        X, y_direction, y_return, feature_names = self.prepare_features(df)
        if X is None:
            print("Could not prepare features")
            return
        
        # Train models
        direction_model = self.train_direction_model(X, y_direction)
        return_model = self.train_return_model(X, y_return)
        
        # Save models
        if direction_model:
            self.models[f'{symbol}_direction'] = direction_model
            joblib.dump(direction_model, f'models/{symbol}_direction_model.pkl')
            
        if return_model:
            self.models[f'{symbol}_return'] = return_model
            joblib.dump(return_model, f'models/{symbol}_return_model.pkl')
        
        # Save feature names
        with open(f'models/{symbol}_features.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        
        print(f"Models trained and saved for {symbol}")
    
    def predict(self, symbol, features):
        """Make predictions using trained models"""
        direction_key = f'{symbol}_direction'
        return_key = f'{symbol}_return'
        
        predictions = {}
        
        if direction_key in self.models:
            direction_prob = self.models[direction_key].predict_proba([features])[0]
            predictions['direction'] = 'UP' if direction_prob[1] > 0.5 else 'DOWN'
            predictions['direction_confidence'] = max(direction_prob)
        
        if return_key in self.models:
            return_pred = self.models[return_key].predict([features])[0]
            predictions['return_1d'] = return_pred
        
        return predictions

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    predictor = PricePredictor()
    
    # Train models for BTC and ETH
    for symbol in ['BTC', 'ETH']:
        try:
            predictor.train_models(symbol)
        except Exception as e:
            print(f"Error training {symbol}: {e}")