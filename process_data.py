from data.s3_manager import S3DataManager
from pipelines.data_cleaner import DataCleaner
from models.finbert_analyzer import FinBERTAnalyzer
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize components
    s3_manager = S3DataManager()
    cleaner = DataCleaner()
    analyzer = FinBERTAnalyzer()
    
    print("Downloading raw data from S3...")
    raw_df = s3_manager.download_raw_data()
    
    if raw_df.empty:
        print("No raw data found!")
        return
    
    print(f"Processing {len(raw_df)} raw data records...")
    
    # Clean the data
    df = cleaner.process_dataframe(raw_df)
    print(f"Cleaned data: {len(df)} records")
    
    # Analyze sentiment
    df = analyzer.process_dataframe(df)
    print("Sentiment analysis complete!")
    
    # Upload processed data
    filename = "processed_data.csv"
    s3_manager.upload_processed_data(df, filename)
    print(f"Uploaded processed data: {filename}")

if __name__ == "__main__":
    main()