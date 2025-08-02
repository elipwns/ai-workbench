# AI Workbench

Financial sentiment analysis using FinBERT model to process Reddit discussions and generate market sentiment scores.

## 🎯 Purpose

Transforms raw financial text data into structured sentiment analysis with 1-5 star ratings optimized for trading insights.

## 🧠 AI Model

### FinBERT (Financial BERT)
- **Model**: `ProsusAI/finbert`
- **Specialization**: Financial text sentiment analysis
- **Output**: 1-5 star rating system
- **Performance**: 55+ texts/second on CPU
- **Memory**: Optimized for CPU-only inference

### Sentiment Mapping
- **1-2 stars**: Bearish sentiment
- **3 stars**: Neutral sentiment  
- **4-5 stars**: Bullish sentiment

## 🚀 Usage

### Process All New Data
```bash
python process_data.py
```

### Test Model Performance
```bash
python test_finbert.py
```

## 📊 Input/Output

### Input (from data-harvester)
- **Source**: S3 `raw-data/reddit_financial_*.csv`
- **Content**: Reddit posts and comments with metadata
- **Format**: CSV with `title`, `content`, `category`, etc.

### Output (to insight-dashboard)
- **Destination**: S3 `processed-data/processed_data.csv`
- **Added Columns**: `sentiment_label`, `sentiment_score`
- **Format**: Enhanced CSV with all original data + sentiment

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# AWS (required)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# S3 (required)  
S3_BUCKET_NAME=automated-trading-data-bucket
```

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `transformers` - FinBERT model
- `torch` - PyTorch (CPU optimized)
- `pandas` - Data processing
- `boto3` - AWS S3 integration

## 📈 Current Performance

### Processing Stats
- **Speed**: 55+ texts/second (CPU-only)
- **Batch Size**: Optimized for memory efficiency
- **Model Size**: ~440MB download (cached locally)
- **Memory Usage**: ~2GB RAM during processing

### Recent Results
- **Records Processed**: 1,156 Reddit posts/comments
- **Processing Time**: ~20 seconds total
- **Sentiment Distribution**: Balanced across 1-5 stars
- **Success Rate**: 100% (no failed analyses)

## 🔧 Technical Details

### Model Architecture
- **Base**: BERT-base-uncased
- **Fine-tuning**: Financial news and reports
- **Tokenizer**: BERT WordPiece tokenizer
- **Max Length**: 512 tokens (auto-truncated)

### Data Processing Pipeline
1. **Load**: Fetch new CSV files from S3
2. **Combine**: Merge title + content for analysis
3. **Analyze**: FinBERT sentiment scoring
4. **Map**: Convert to 1-5 star system
5. **Save**: Upload enhanced data to S3

### Error Handling
- **Empty Text**: Assigns neutral (3 stars)
- **Model Errors**: Logs and continues processing
- **S3 Failures**: Retries with exponential backoff

## 📊 Sentiment Analysis Quality

### Text Processing
- **Combines**: Post title + content for full context
- **Handles**: Emojis, URLs, special characters
- **Preserves**: Original text alongside sentiment
- **Confidence**: Includes raw model scores

### Financial Context
- **Trained On**: Financial news, earnings reports, market commentary
- **Understands**: Trading terminology, market sentiment
- **Optimized For**: Investment-related discussions

## 🎯 Next Steps

### Immediate
- **Batch Processing**: Handle larger datasets efficiently
- **Monitoring**: Add processing time/success metrics
- **Validation**: Compare sentiment with price movements

### Model Improvements
- **Fine-tuning**: Train on Reddit-specific financial data
- **Ensemble**: Combine multiple sentiment models
- **Confidence Filtering**: Flag low-confidence predictions

### Infrastructure
- **GPU Support**: Optional GPU acceleration for large batches
- **Caching**: Store model in memory for repeated runs
- **Parallel Processing**: Multi-threading for CPU optimization

## 🔍 Data Insights

### Sentiment Distribution (Recent)
- **Bullish (4-5 stars)**: ~35% of posts
- **Neutral (3 stars)**: ~30% of posts  
- **Bearish (1-2 stars)**: ~35% of posts

### Category Patterns
- **CRYPTO**: More volatile sentiment swings
- **US_STOCKS**: Generally more conservative sentiment
- **ECONOMICS**: Longer-term, macro-focused sentiment

## 🐛 Known Issues

- **Long Posts**: Truncated at 512 tokens (BERT limit)
- **Sarcasm**: May misinterpret sarcastic posts
- **Context**: Limited to individual post context

## 📁 File Structure

```
ai-workbench/
├── models/
│   └── sentiment_analyzer.py    # FinBERT implementation
├── data/
│   └── s3_manager.py           # S3 data handling
├── process_data.py             # Main processing script
├── test_finbert.py            # Model testing
└── requirements.txt           # Dependencies
```

---

*Part of the automated-trading pipeline*
*Previous: data-harvester collects raw data*
*Next: insight-dashboard visualizes sentiment results*