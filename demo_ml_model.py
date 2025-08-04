#!/usr/bin/env python3
"""
Demo ML Model - Simple price prediction using available data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.s3_manager import S3DataManager

def create_demo_dataset():
    """Create demo dataset with synthetic features"""
    np.random.seed(42)
    
    # Generate 100 days of synthetic data ending today
    from datetime import datetime
    end_date = datetime.now().date()
    dates = pd.date_range(end=end_date, periods=100, freq='D')
    
    data = []
    price = 50000  # Starting BTC price
    
    for i, date in enumerate(dates):
        # Synthetic sentiment (random walk)
        sentiment = 50 + np.random.normal(0, 10)
        sentiment = max(0, min(100, sentiment))
        
        # Price influenced by sentiment + noise
        price_change = (sentiment - 50) * 0.02 + np.random.normal(0, 2)
        price = max(price * (1 + price_change/100), 1000)
        
        # Technical indicators
        sma_7 = price * (1 + np.random.normal(0, 0.01))
        rsi = 50 + np.random.normal(0, 15)
        rsi = max(0, min(100, rsi))
        
        # Target: will price go up tomorrow?
        if i < len(dates) - 1:
            tomorrow_change = np.random.normal((sentiment - 50) * 0.01, 1)
            target = 1 if tomorrow_change > 0 else 0
        else:
            target = np.nan
        
        data.append({
            'date': date,
            'price': price,
            'sentiment_mean': sentiment,
            'sma_7': sma_7,
            'rsi': rsi,
            'volume_ratio': 1 + np.random.normal(0, 0.5),
            'target_direction_1d': target
        })
    
    return pd.DataFrame(data)

def train_demo_model():
    """Train a simple demo model"""
    print("Creating demo dataset...")
    df = create_demo_dataset()
    
    # Prepare features
    feature_cols = ['sentiment_mean', 'sma_7', 'rsi', 'volume_ratio']
    X = df[feature_cols].fillna(0)
    y = df['target_direction_1d'].fillna(0).astype(int)
    
    # Remove last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    print(f"Training with {len(X)} samples, {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    # Make sample predictions
    print("\nSample Predictions:")
    for i in range(3):
        features = X_test.iloc[i].values
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]
        actual = y_test.iloc[i]
        
        print(f"Sample {i+1}: Predicted={pred} (confidence={max(prob):.3f}), Actual={actual}")
    
    return model

if __name__ == "__main__":
    model = train_demo_model()