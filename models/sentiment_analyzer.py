from transformers import pipeline
import pandas as pd
import torch
from typing import List

class SentimentAnalyzer:
    def __init__(self):
        # Using CPU for sentiment analysis (RTX 5090 not yet supported by PyTorch)
        device = -1
        print(f"Using device: CPU")
        
        # Use financial sentiment model (FinBERT would be better but requires more setup)
        self.classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device
        )
    
    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze sentiment for a batch of texts"""
        # Truncate texts to avoid token limits
        truncated_texts = [text[:512] for text in texts]
        results = self.classifier(truncated_texts)
        return results
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """Add sentiment analysis to DataFrame"""
        texts = df[text_column].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.analyze_batch(batch)
            all_results.extend(batch_results)
        
        # Add results to DataFrame
        df['sentiment_label'] = [r['label'] for r in all_results]
        df['sentiment_score'] = [r['score'] for r in all_results]
        
        return df