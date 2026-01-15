# Fake News Detection System

A machine learning-based web application that detects fake news using Random Forest classifier and NLP techniques. Built with Streamlit for an intuitive user interface.

[![Live App](https://img.shields.io/badge/Streamlit-Live-brightgreen)](https://fake-news-detection-system-lfmyo8mzhohrqgyjqdu7lh.streamlit.app/)

Try the app online to analyze news articles or social media posts in real-time and get FAKE/REAL predictions instantly.
## 🚀 Features

- **Text Analysis**: Analyze news articles, social media posts, or any text content
- **URL Analysis**: Extract and analyze content from news article URLs
- **Real-time Prediction**: Get instant FAKE/REAL predictions with credibility scores
- **Detailed Analysis**: View breakdowns of factual accuracy, source credibility, bias, and clickbait detection
- **History Tracking**: Save and review past analysis results
- **Report Management**: Download and manage analysis reports in JSON format

## 📋 Requirements

- Python 3.8+
- Required packages (see `requirements.txt` or install manually):
  - streamlit
  - pandas
  - scikit-learn
  - joblib
  - beautifulsoup4
  - requests
  - numpy

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake_news_testing_10_01_2026
```

2. Install dependencies:
```bash
pip install streamlit pandas scikit-learn joblib beautifulsoup4 requests numpy
```

3. Prepare your training data:
   - Place `Fake.csv` and `True.csv` in the `trainDataSet/` folder
   - The model will automatically train on first run if `model.joblib` doesn't exist

4. Run the application:
```bash
streamlit run fakeNews.py
```

## 📁 Project Structure

```
fake_news_testing_10_01_2026/
├── fakeNews.py              # Main Streamlit application
├── analyzer.py              # Analysis logic and ML model integration
├── scikitLearn.py           # ML model training and prediction
├── utils.py                 # Utility functions (URL extraction, display)
├── train_model_once.py      # Script to train model manually
├── check_accuracy.py        # Script to test model accuracy
├── visualize_top_features.py # Feature importance visualization
├── trainDataSet/            # Training data
│   ├── Fake.csv
│   └── True.csv
├── testDataSet/             # Test data (optional)
│   └── news.csv
├── reports/                 # Generated analysis reports
├── model.joblib             # Trained model (generated)
└── vectorizer.joblib        # TF-IDF vectorizer (generated)
```

## 🎯 Usage

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run fakeNews.py
```

2. Open your browser to the URL shown (usually `http://localhost:8501`)

3. Choose input method:
   - **Text Input**: Paste news text directly
   - **URL Analysis**: Enter a news article URL

4. Click "Analyze Text" or "Analyze URL"

5. View results:
   - Credibility Score (0-100%)
   - ML Model Prediction (FAKE/REAL)
   - Detailed analysis breakdown

### Training the Model

The model trains automatically on first run. To retrain manually:

```bash
python train_model_once.py
```

Or use the scikitLearn module directly:

```bash
python scikitLearn.py
```

### Checking Model Accuracy

Test the model's performance:

```bash
python check_accuracy.py
```

## 🔧 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 10,000 TF-IDF features
- **N-grams**: 1-3 word combinations (unigrams, bigrams, trigrams)
- **Accuracy**: ~97% on validation set, ~92% on test dataset
- **Text Preprocessing**: Lowercasing, URL removal, punctuation normalization

## 📊 How It Works

1. **Text Preprocessing**: Cleans and normalizes input text
2. **Validation**: Checks for gibberish or invalid content
3. **Vectorization**: Converts text to numerical features using TF-IDF
4. **Prediction**: Random Forest model predicts FAKE/REAL
5. **Credibility Score**: Calculated from model's probability confidence
6. **Analysis**: Generates detailed breakdown of various factors

## 🐛 Troubleshooting

- **Model not found**: Run `python train_model_once.py` to generate model files
- **Low accuracy**: Ensure training data is properly formatted in `trainDataSet/`
- **Import errors**: Install all required packages with `pip install -r requirements.txt`

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
