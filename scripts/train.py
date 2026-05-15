"""
scripts/train.py
----------------
Full training pipeline — downloads data, engineers features,
trains Random Forest, evaluates, and saves model to models/
"""

import os
import joblib
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Config ───────────────────────────────────────────────────────────────────
TICKERS    = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'JNJ']
START_DATE = "2020-01-01"
END_DATE   = "2024-01-01"
MODEL_PATH = "models/stock_model_1.pkl"

FEATURES = [
    'Return', 'Volatility', 'Momentum',
    'MA_10_slope', 'MA_50_slope',
    'Price_vs_MA10', 'Price_vs_MA50',
    'Volume_change', 'High_Low_range',
]

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df.columns = df.columns.droplevel(1)
    except Exception:
        pass

    df['Return']         = df['Close'].pct_change()
    df['MA_10']          = df['Close'].rolling(10).mean()
    df['MA_50']          = df['Close'].rolling(50).mean()
    df['Volatility']     = df['Return'].rolling(10).std()
    df['Momentum']       = df['Close'] - df['MA_10']
    df['MA_10_slope']    = df['MA_10'].pct_change()
    df['MA_50_slope']    = df['MA_50'].pct_change()
    df['Price_vs_MA10']  = df['Close'] / df['MA_10']
    df['Price_vs_MA50']  = df['Close'] / df['MA_50']
    df['Volume_change']  = df['Volume'].pct_change()
    df['High_Low_range'] = (df['High'] - df['Low']) / df['Close']
    df['Target']         = (df['Return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df


# ── Download & combine data ───────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    all_data = []
    for ticker in TICKERS:
        print(f"  Downloading {ticker}...")
        raw      = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        featured = build_features(raw)
        all_data.append(featured)
    combined = pd.concat(all_data)
    print(f"  Total rows: {combined.shape[0]}")
    return combined


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n[1/4] Downloading data...")
    df = load_data()

    print("\n[2/4] Preparing features...")
    X = df[FEATURES]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("\n[3/4] Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n[4/4] Evaluating...")
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
