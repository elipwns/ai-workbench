from data.s3_manager import S3DataManager
from pipelines.data_cleaner import DataCleaner
from models.sentiment_analyzer import SentimentAnalyzer
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize components
    s3_manager = S3DataManager()
    cleaner = DataCleaner()
    analyzer = SentimentAnalyzer()
    
    print("Downloading raw data from S3...")
    raw_data = s3_manager.download_raw_data()
    
    if not raw_data:
        print("No raw data found!")
        return
    
    print(f"Processing {len(raw_data)} raw data files...")
    
    # Clean the data
    df = cleaner.process_scraped_data(raw_data)
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