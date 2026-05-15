# 📈 StockSense AI — Stock Direction Predictor

A machine learning web application that predicts whether a stock price will go **UP** or **DOWN** the next trading day, using a Random Forest classifier trained on 9 engineered technical features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🖥️ Demo

> Run locally with `streamlit run app.py`

---

## 🧠 How It Works

1. **Data Collection** — Historical OHLCV data fetched via `yfinance` for 6 stocks
2. **Feature Engineering** — 9 technical indicators computed from raw price data
3. **Model Training** — Random Forest Classifier trained on ~9,000 rows across 6 stocks
4. **Live Prediction** — App fetches latest 6 months of data in real time and predicts next-day direction

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `Return` | Daily % price change |
| `Volatility` | 10-day rolling standard deviation of returns |
| `Momentum` | Close price minus 10-day moving average |
| `MA_10_slope` | Rate of change of 10-day moving average |
| `MA_50_slope` | Rate of change of 50-day moving average |
| `Price_vs_MA10` | Close / MA10 ratio |
| `Price_vs_MA50` | Close / MA50 ratio |
| `Volume_change` | Daily % change in trading volume |
| `High_Low_range` | (High - Low) / Close — intraday volatility |

---

## 🏦 Supported Stocks

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet (Google) | Technology |
| TSLA | Tesla Inc. | EV / Energy |
| JPM | JPMorgan Chase | Financials |
| JNJ | Johnson & Johnson | Healthcare |

---

## 📁 Project Structure

```
stock-direction-predictor/
│
├── app.py                      # Streamlit web application
│
├── inference/
│   └── predict.py              # Clean prediction pipeline function
│
├── scripts/
│   └── train.py                # Full model training script
│
├── models/
│   └── stock_model_1.pkl       # Saved trained model
│
├── notebooks/
│   └── model_training.ipynb    # Jupyter notebook (EDA + training)
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/stock-direction-predictor.git
cd stock-direction-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Retrain the model (optional)
```bash
python scripts/train.py
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Test Accuracy | ~53% |
| Training Stocks | 6 |
| Training Rows | ~9,000 |
| Features | 9 |

> **Note:** In financial ML, even a 53% accurate model can be profitable over thousands of trades. Hedge funds pay millions for a consistent 2-3% edge over random.

---

## ⚠️ Disclaimer

This project is built **for educational purposes only**. It is not financial advice. Past patterns do not guarantee future results. Always do your own research before making any investment decisions.

---

## 👨‍💻 Author

Built with ❤️ as a portfolio project to demonstrate end-to-end ML engineering skills:
- Data collection & EDA
- Feature engineering
- Model training & evaluation
- Web app deployment

---

## 📄 License

MIT License — free to use and modify.
