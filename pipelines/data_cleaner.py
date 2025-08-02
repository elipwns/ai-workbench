import pandas as pd
import re
from typing import List, Dict

class DataCleaner:
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def process_scraped_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Convert raw scraped data to clean DataFrame"""
        processed_records = []
        
        for batch in raw_data:
            for result in batch.get('results', []):
                record = {
                    'url': result.get('url', ''),
                    'title': self.clean_text(result.get('title', '')),
                    'content': self.clean_text(result.get('text_content', '')),
                    'scraped_at': result.get('scraped_at', '')
                }
                processed_records.append(record)
        
        df = pd.DataFrame(processed_records)
        # Remove duplicates and empty content
        df = df.drop_duplicates(subset=['url'])
        df = df[df['content'].str.len() > 50]  # Filter out very short content
        
        return df