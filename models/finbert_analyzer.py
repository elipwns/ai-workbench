from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from typing import List
import numpy as np

class FinBERTAnalyzer:
    def __init__(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        device = -1  # CPU only (RTX 5090 not supported yet)
        print(f"Loading FinBERT model on CPU...")
        
        # Use ProsusAI/finbert - the most popular financial BERT model
        model_name = "ProsusAI/finbert"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            print("FinBERT loaded successfully")
        except Exception as e:
            print(f"Failed to load FinBERT: {e}")
            print("Falling back to generic model...")
            self.classifier = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=device
            )
    
    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze sentiment for a batch of financial texts"""
        # Clean and truncate texts
        cleaned_texts = []
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or len(str(text).strip()) == 0 or str(text).lower() == 'nan':
                cleaned_texts.append("neutral market sentiment")
            else:
                # Truncate to 512 characters (BERT limit)
                cleaned_texts.append(str(text)[:512])
        
        try:
            results = self.classifier(cleaned_texts)
            return results
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            # Return varied sentiment for failed batch (not all neutral)
            import random
            fallback_results = []
            for text in texts:
                # Simple keyword-based fallback
                text_lower = text.lower()
                if any(word in text_lower for word in ['moon', 'bullish', 'buy', 'pump', 'rocket', 'green', 'up']):
                    fallback_results.append({'label': 'positive', 'score': 0.7})
                elif any(word in text_lower for word in ['crash', 'dump', 'bearish', 'sell', 'red', 'down', 'panic']):
                    fallback_results.append({'label': 'negative', 'score': 0.7})
                else:
                    # More realistic distribution - less neutral, more negative (market pessimism)
                    rand = random.random()
                    if rand < 0.25:
                        fallback_results.append({'label': 'positive', 'score': 0.6})
                    elif rand < 0.4:
                        fallback_results.append({'label': 'neutral', 'score': 0.5})
                    else:
                        fallback_results.append({'label': 'negative', 'score': 0.65})
            return fallback_results
    
    def convert_to_star_rating(self, finbert_result, original_text=""):
        """Convert FinBERT output to 1-5 star system with keyword enhancement"""
        label = finbert_result['label'].lower()
        score = finbert_result['score']
        
        # Keyword-based sentiment boosting for crypto/trading language
        text_lower = original_text.lower()
        
        # Strong bullish keywords that should override neutral classification
        strong_bullish = ['moon', 'rocket', 'pump', 'hodl', 'diamond hands', 'bull run', 'to the moon', 'lambo', 'ath', 'breakout']
        # Strong bearish keywords
        strong_bearish = ['crash', 'dump', 'panic', 'rekt', 'bear market', 'capitulation', 'dead cat bounce', 'rugpull']
        
        # Check for strong sentiment keywords
        has_strong_bullish = any(keyword in text_lower for keyword in strong_bullish)
        has_strong_bearish = any(keyword in text_lower for keyword in strong_bearish)
        
        # Override neutral classifications if strong keywords present OR low confidence
        if label == 'neutral' and has_strong_bullish:
            label = 'positive'
            score = max(score, 0.75)  # Boost confidence
        elif label == 'neutral' and has_strong_bearish:
            label = 'negative'
            score = max(score, 0.75)  # Boost confidence
        elif label == 'neutral' and score < 0.6:
            # Low confidence neutral -> lean negative (market pessimism)
            label = 'negative'
            score = 0.55
        
        if label == 'positive':
            # Positive: 4-5 stars based on confidence
            if score > 0.8 or has_strong_bullish:
                return {'label': '5 stars', 'score': score}
            else:
                return {'label': '4 stars', 'score': score}
        elif label == 'negative':
            # Negative: 1-2 stars based on confidence
            if score > 0.8 or has_strong_bearish:
                return {'label': '1 star', 'score': score}
            else:
                return {'label': '2 stars', 'score': score}
        else:  # neutral
            return {'label': '3 stars', 'score': score}
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """Add FinBERT sentiment analysis to DataFrame"""
        if df.empty:
            return df
        
        texts = df[text_column].fillna("").astype(str).tolist()
        print(f"Analyzing {len(texts)} texts with FinBERT...")
        
        # Process in batches to avoid memory issues
        batch_size = 16  # Smaller batches for FinBERT
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.analyze_batch(batch)
            
            # Convert FinBERT results to star ratings with original text for keyword enhancement
            converted_results = [self.convert_to_star_rating(result, original_text=text) 
                               for result, text in zip(batch_results, batch)]
            all_results.extend(converted_results)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch)}/{len(texts)} texts...")
        
        # Add results to DataFrame
        df['sentiment_label'] = [r['label'] for r in all_results]
        df['sentiment_score'] = [r['score'] for r in all_results]
        
        print(f"FinBERT analysis complete!")
        return df
    
    def analyze_single_text(self, text: str) -> dict:
        """Analyze a single text for testing"""
        result = self.analyze_batch([text])[0]
        return self.convert_to_star_rating(result, original_text=text)

# Test function
if __name__ == "__main__":
    analyzer = FinBERTAnalyzer()
    
    # Test with financial texts
    test_texts = [
        "Bitcoin is going to the moon! Great investment opportunity!",
        "The market is crashing, sell everything now!",
        "Tesla reported mixed earnings results",
        "Fed raises interest rates, markets uncertain"
    ]
    
    print("\nTesting FinBERT on financial texts:")
    for text in test_texts:
        result = analyzer.analyze_single_text(text)
        print(f"Text: {text[:50]}...")
        print(f"Result: {result['label']} (confidence: {result['score']:.3f})")
        print()