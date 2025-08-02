import boto3
import json
import pandas as pd
from typing import List, Dict
import os

class S3DataManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def download_raw_data(self, prefix='raw-data/') -> pd.DataFrame:
        """Download all raw data from S3 and return as DataFrame"""
        objects = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, 
            Prefix=prefix
        )
        
        all_data = []
        for obj in objects.get('Contents', []):
            key = obj['Key']
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, 
                Key=key
            )
            
            # Handle both CSV and JSON files
            if key.endswith('.csv'):
                from io import StringIO
                csv_content = response['Body'].read().decode('utf-8')
                df = pd.read_csv(StringIO(csv_content))
                all_data.append(df)
            elif key.endswith('.json'):
                content = json.loads(response['Body'].read())
                if isinstance(content, list):
                    df = pd.DataFrame(content)
                else:
                    df = pd.DataFrame([content])
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def upload_processed_data(self, data: pd.DataFrame, filename: str):
        """Upload processed data back to S3"""
        key = f"processed-data/{filename}"
        csv_buffer = data.to_csv(index=False)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=csv_buffer,
            ContentType='text/csv'
        )
        print(f"Processed data uploaded to s3://{self.bucket_name}/{key}")