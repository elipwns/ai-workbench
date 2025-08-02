# AI Workbench

Financial sentiment analysis and data processing pipeline for automated trading system.

## Part of 3-Repo System

- **[Data Harvester](https://github.com/elipwns/data-harvester)** - Web scraping pipeline
- **[AI Workbench](https://github.com/elipwns/ai-workbench)** ← You are here
- **[Insight Dashboard](https://github.com/elipwns/insight-dashboard)** - Data visualization

## Current Status

✅ **Working Components:**
- Financial sentiment analysis using multilingual BERT model
- S3 data management (download/upload)
- Data cleaning pipeline
- CPU-optimized processing (55+ texts/second)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS settings
   ```

3. **Test sentiment analysis:**
   ```bash
   python3 test_finbert.py
   ```

4. **Process data from S3:**
   ```bash
   python3 process_data.py
   ```

## Architecture

- **`models/sentiment_analyzer.py`** - Financial sentiment analysis (1-5 star rating)
- **`data/s3_manager.py`** - S3 data operations
- **`pipelines/data_cleaner.py`** - Text preprocessing
- **`process_data.py`** - Main processing pipeline

## Data Flow

1. Raw data from [Data Harvester](https://github.com/elipwns/data-harvester) → S3 `raw-data/`
2. AI processing → Sentiment analysis + cleaning
3. Processed data → S3 `processed-data/`
4. Dashboard consumption via [Insight Dashboard](https://github.com/elipwns/insight-dashboard)

## GPU Support

Currently running on CPU due to RTX 5090 compatibility. GPU support will be enabled when PyTorch adds sm_120 capability support.

## Requirements

- Python 3.10+
- AWS credentials configured
- S3 bucket: `automated-trading-data-bucket`
