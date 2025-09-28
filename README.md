# Country Region Classifier

An AI-powered tool that automatically classifies countries by their geographical regions using Google's Gemini AI.

## 🌍 Data Source

 - dataset: ["BI intro to data cleaning eda and machine learning"]
 - Url: (https://www.kaggle.com/datasets/walekhwatlphilip/intro-to-data-cleaning-eda-and-machine-learning) 
 - Author: walekhwatlphilip.

## ✨ Features

- Batch processing for efficient API usage
- Parallel processing for faster execution  
- Checkpoint system to resume interrupted sessions
- Automatic encoding detection for CSV files
- Error recovery with retry logic
- Real-time progress tracking

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Key

1. Get your Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Open the `.env` file in this project
3. Replace `your_actual_api_key_here` with your real API key
4. Save the file

### 3. Get Dataset

1. Download the dataset from the Kaggle link above
2. Extract the CSV file  
3. Rename it to `bi.csv` and place in the project folder
4. Make sure it has a column named "country"

### 4. Run the Script

```bash
python bi.py
```

## 📊 Output

The script will create a `complete.csv` file with your original data plus a new "region" column:

- Africa
- Asia
- The Caribbean  
- Central America
- Europe
- North America
- Oceania
- South America

## ⚙️ Configuration

You can modify these settings in the code:

- `batch_size`: How many countries to process at once (default: 8)
- `max_workers`: Number of parallel threads (default: 3)
- Input/output filenames

## 🔧 Troubleshooting

**"GOOGLE_API_KEY not found"**: Make sure you edited the .env file with your real API key

**"FileNotFoundError: bi.csv"**: Download the dataset from Kaggle and name it bi.csv

**"UnicodeDecodeError"**: The script handles this automatically by trying different encodings

**API Rate Limits**: Reduce batch_size and max_workers in the script

## 📜 License

MIT License - feel free to use and modify this code.

## 🙏 Credits

- Dataset by [walekhwatlphilip](https://www.kaggle.com/walekhwatlphilip) on Kaggle
- Uses Google's Gemini AI API

- Built with Python and pandas
