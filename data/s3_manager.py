import boto3
import json
import pandas as pd
from typing import List, Dict
import os

class S3DataManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def download_raw_data(self, prefix='raw-data/') -> List[Dict]:
        """Download all raw data from S3"""
        objects = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, 
            Prefix=prefix
        )
        
        data = []
        for obj in objects.get('Contents', []):
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, 
                Key=obj['Key']
            )
            content = json.loads(response['Body'].read())
            data.append(content)
        
        return data
    
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