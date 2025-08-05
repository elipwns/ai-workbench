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
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame with Reddit or Bluesky data"""
        # Handle different data sources based on platform field
        def create_content(row):
            if row.get('platform') == 'bluesky':
                # Bluesky data - use text field directly
                text = row.get('text', '')
                if pd.isna(text):
                    text = ''
                return self.clean_text(str(text))
            else:
                # Reddit data - combine title and content
                title = row.get('title', '')
                content = row.get('content', '')
                if pd.isna(title):
                    title = ''
                if pd.isna(content):
                    content = ''
                return self.clean_text(f"{title} {content}".strip())
        
        df['content'] = df.apply(create_content, axis=1)
        
        # Remove duplicates and empty content
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        df = df[df['content'].str.len() > 10]  # Filter out very short content
        
        return df