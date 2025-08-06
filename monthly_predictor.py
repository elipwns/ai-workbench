#!/usr/bin/env python3
"""
Monthly Prediction System
Generates monthly price predictions with confidence bands and tracks performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.s3_manager import S3DataManager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

class MonthlyPredictor:
    def __init__(self):
        self.s3_manager = S3DataManager()
        self.predictions_file = "monthly_predictions.json"
        
    def load_predictions_history(self):
        """Load historical predictions from S3"""
        try:
            response = self.s3_manager.s3_client.get_object(
                Bucket=self.s3_manager.bucket_name,
                Key=f"predictions/{self.predictions_file}"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except:
            return {"predictions": [], "performance": []}
    
    def save_predictions_history(self, data):
        """Save predictions history to S3"""
        self.s3_manager.s3_client.put_object(
            Bucket=self.s3_manager.bucket_name,
            Key=f"predictions/{self.predictions_file}",
            Body=json.dumps(data, indent=2, default=str),
            ContentType='application/json'
        )
    
    def get_current_price(self, symbol='BTC'):
        """Get current price for symbol"""
        try:
            # Get most recent price data
            objects = self.s3_manager.s3_client.list_objects_v2(
                Bucket=self.s3_manager.bucket_name,
                Prefix="raw-data/price_data_"
            )
            
            if not objects.get('Contents'):
                return None
            
            # Get most recent file
            latest = sorted(objects['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
            
            response = self.s3_manager.s3_client.get_object(
                Bucket=self.s3_manager.bucket_name,
                Key=latest['Key']
            )
            
            from io import StringIO
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            symbol_data = df[df['symbol'] == symbol]
            if not symbol_data.empty:
                return float(symbol_data.iloc[-1]['price'])
            
            return None
            
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
    
    def load_ml_features(self, symbol='BTC'):
        """Load ML features if available"""
        try:
            objects = self.s3_manager.s3_client.list_objects_v2(
                Bucket=self.s3_manager.bucket_name,
                Prefix=f"raw-data/ml_features_{symbol}_"
            )
            
            if not objects.get('Contents'):
                return pd.DataFrame()
            
            # Get most recent ML features file
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
            
            print(f"Loaded ML features: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"No ML features available: {e}")
            return pd.DataFrame()
    
    def create_enhanced_model(self, symbol='BTC'):
        """Create enhanced prediction model using ML features"""
        try:
            # Try to load ML features first
            ml_df = self.load_ml_features(symbol)
            
            if not ml_df.empty and len(ml_df) >= 10:
                print("Using enhanced ML model with technical indicators")
                return self.create_ml_model(ml_df, symbol)
            else:
                print("Using simple sentiment model (insufficient ML data)")
                return self.create_simple_model(symbol)
                
        except Exception as e:
            print(f"Error creating enhanced model: {e}")
            return self.create_simple_model(symbol)
    
    def create_ml_model(self, ml_df, symbol='BTC'):
        """Create ML-based prediction using technical indicators"""
        try:
            # Get latest features
            latest = ml_df.iloc[-1]
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                return None, None
            
            # Enhanced feature vector
            features = {
                'sentiment_mean': latest.get('sentiment_mean', 3.0),
                'sentiment_momentum_7d': latest.get('sentiment_momentum_7d', 0),
                'rsi': latest.get('rsi', 50),
                'macd': latest.get('macd', 0),
                'bb_position': latest.get('bb_position', 0.5),
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'price_change_7d': latest.get('price_change_7d', 0)
            }
            
            # Enhanced prediction logic
            base_prediction = current_price
            
            # Sentiment factor (stronger weight)
            sentiment_factor = (features['sentiment_mean'] - 3) * 0.06
            sentiment_momentum = features['sentiment_momentum_7d'] * 0.04
            
            # Technical factors
            rsi_factor = 0 if pd.isna(features['rsi']) else (features['rsi'] - 50) / 1000  # RSI momentum
            macd_factor = 0 if pd.isna(features['macd']) else features['macd'] * 0.001  # MACD signal
            bb_factor = 0 if pd.isna(features['bb_position']) else (features['bb_position'] - 0.5) * 0.02  # Bollinger position
            
            # Volume factor
            volume_factor = 0 if pd.isna(features['volume_ratio']) else (features['volume_ratio'] - 1) * 0.01
            
            # Price momentum
            momentum_factor = 0 if pd.isna(features['price_change_7d']) else features['price_change_7d'] * 0.1
            
            # Combined prediction
            total_factor = (
                sentiment_factor + sentiment_momentum +  # 60% weight
                rsi_factor + macd_factor + bb_factor +   # 30% weight  
                volume_factor + momentum_factor * 0.5    # 10% weight
            )
            
            prediction = base_prediction * (1 + total_factor)
            
            # Dynamic confidence based on volatility
            volatility = abs(features['price_change_7d']) if not pd.isna(features['price_change_7d']) else 0.05
            confidence = max(0.06, min(0.12, volatility))  # 6-12% based on recent volatility
            
            print(f"Enhanced model factors: sentiment={sentiment_factor:.3f}, technical={rsi_factor+macd_factor+bb_factor:.3f}, volume={volume_factor:.3f}")
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error in ML model: {e}")
            return None, None
    
    def create_simple_model(self, symbol='BTC'):
        """Create simple prediction model based on available data"""
        try:
            # Load recent sentiment and price data
            processed_df = self.s3_manager.download_raw_data()
            
            if processed_df.empty:
                return None, None
            
            # Simple features: sentiment trend, price momentum
            if 'sentiment_label' in processed_df.columns:
                # Convert sentiment to numeric
                sentiment_map = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5}
                processed_df['sentiment_numeric'] = processed_df['sentiment_label'].map(sentiment_map)
                
                # Daily aggregation
                processed_df['date'] = pd.to_datetime(processed_df['timestamp']).dt.date
                daily_sentiment = processed_df.groupby('date')['sentiment_numeric'].mean()
                
                # Get price data
                current_price = self.get_current_price(symbol)
                if not current_price:
                    return None, None
                
                # Simple trend features
                recent_sentiment = daily_sentiment.tail(7).mean()  # Last week average
                sentiment_trend = daily_sentiment.tail(3).mean() - daily_sentiment.tail(7).mean()
                
                features = np.array([[recent_sentiment, sentiment_trend, current_price]])
                
                # Simple model: assume some correlation between sentiment and price
                base_prediction = current_price
                sentiment_factor = (recent_sentiment - 3) * 0.05  # 5% per sentiment point above/below neutral
                trend_factor = sentiment_trend * 0.1
                
                prediction = base_prediction * (1 + sentiment_factor + trend_factor)
                confidence = 0.08  # 8% confidence band - tight for clear evaluation
                
                return prediction, confidence
            
            return None, None
            
        except Exception as e:
            print(f"Error creating model: {e}")
            return None, None
    
    def generate_monthly_prediction(self, symbol='BTC'):
        """Generate prediction for next month"""
        current_date = datetime.now()
        
        # Always predict for next month
        prediction_month = current_date.replace(day=1) + timedelta(days=32)
        prediction_month = prediction_month.replace(day=1)  # Next month
        
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"Could not get current price for {symbol}")
            return None
        
        # Generate prediction using enhanced model (falls back to simple if no ML data)
        predicted_price, confidence = self.create_enhanced_model(symbol)
        
        if predicted_price is None:
            # Fallback: use current price with small random walk
            predicted_price = current_price * (1 + np.random.normal(0, 0.05))
            confidence = 0.08
        
        # Calculate confidence bands
        upper_band = predicted_price * (1 + confidence)
        lower_band = predicted_price * (1 - confidence)
        
        prediction = {
            'symbol': symbol,
            'prediction_date': current_date.isoformat(),
            'target_month': prediction_month.strftime('%Y-%m'),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence_band': confidence,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'prediction_change_pct': ((predicted_price - current_price) / current_price) * 100,
            'model_version': 'enhanced_v1' if confidence != 0.08 else 'simple_v1',
            'features_used': ['sentiment_trend', 'technical_indicators', 'volume_data'] if confidence != 0.08 else ['sentiment_trend', 'price_momentum']
        }
        
        return prediction
    
    def evaluate_last_month_prediction(self, symbol='BTC'):
        """Evaluate how last month's prediction performed"""
        history = self.load_predictions_history()
        
        if not history['predictions']:
            return None
        
        # Find last month's prediction
        last_month = datetime.now().replace(day=1) - timedelta(days=1)
        target_month = last_month.strftime('%Y-%m')
        
        last_prediction = None
        for pred in history['predictions']:
            if pred['symbol'] == symbol and pred['target_month'] == target_month:
                last_prediction = pred
                break
        
        if not last_prediction:
            return None
        
        # Get actual price for that month
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        # Calculate performance metrics
        predicted = last_prediction['predicted_price']
        actual = current_price
        error_pct = abs((predicted - actual) / actual) * 100
        
        # Check if actual price was within confidence band
        within_band = (last_prediction['lower_band'] <= actual <= last_prediction['upper_band'])
        
        # Assign performance rating (stricter for tighter bands)
        if within_band and error_pct < 3:
            rating = 'Excellent'
        elif within_band and error_pct < 6:
            rating = 'Good'
        elif within_band:
            rating = 'Fair'
        elif error_pct < 10:
            rating = 'Poor'
        else:
            rating = 'Failed'
        
        performance = {
            'symbol': symbol,
            'target_month': target_month,
            'predicted_price': predicted,
            'actual_price': actual,
            'error_pct': error_pct,
            'within_confidence_band': within_band,
            'rating': rating,
            'evaluation_date': datetime.now().isoformat()
        }
        
        return performance
    
    def run_monthly_prediction_cycle(self, symbols=['BTC']):
        """Run complete monthly prediction cycle"""
        print("=== Monthly Prediction Cycle ===")
        
        history = self.load_predictions_history()
        
        # Evaluate last month's predictions
        for symbol in symbols:
            performance = self.evaluate_last_month_prediction(symbol)
            if performance:
                history['performance'].append(performance)
                print(f"✅ {symbol} last month: {performance['rating']} ({performance['error_pct']:.1f}% error)")
        
        # Generate new predictions for next month
        for symbol in symbols:
            prediction = self.generate_monthly_prediction(symbol)
            if prediction:
                # Check if prediction already exists for this symbol and target month
                existing_prediction = None
                for i, existing in enumerate(history['predictions']):
                    if (existing['symbol'] == prediction['symbol'] and 
                        existing['target_month'] == prediction['target_month']):
                        existing_prediction = i
                        break
                
                if existing_prediction is not None:
                    # Update existing prediction
                    history['predictions'][existing_prediction] = prediction
                    print(f"Updated prediction {symbol} next month: ${prediction['predicted_price']:.0f} (+/-{prediction['confidence_band']*100:.0f}%)")
                else:
                    # Add new prediction
                    history['predictions'].append(prediction)
                    print(f"New prediction {symbol} next month: ${prediction['predicted_price']:.0f} (+/-{prediction['confidence_band']*100:.0f}%)")
        
        # Save updated history
        self.save_predictions_history(history)
        print(f"Saved predictions to S3: {self.predictions_file}")
        
        return history

if __name__ == "__main__":
    predictor = MonthlyPredictor()
    predictor.run_monthly_prediction_cycle(['BTC'])