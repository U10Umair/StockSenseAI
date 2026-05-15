"""
inference/predict.py
--------------------
Clean prediction pipeline — loads model and returns prediction for any ticker.
Used by app.py and can also be called standalone.
"""

import joblib
import yfinance as yf
import pandas as pd

FEATURES = [
    'Return', 'Volatility', 'Momentum',
    'MA_10_slope', 'MA_50_slope',
    'Price_vs_MA10', 'Price_vs_MA50',
    'Volume_change', 'High_Low_range',
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all features from raw OHLCV dataframe."""
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
    df.dropna(inplace=True)
    return df


def predict(ticker: str, model_path: str = "models/stock_model_1.pkl") -> dict:
    """
    Fetch latest data for ticker and return prediction.

    Returns:
        dict with keys:
            - ticker (str)
            - prediction (int): 1=UP, 0=DOWN
            - signal (str): 'UP' or 'DOWN'
            - confidence (float): 0-100
            - prob_up (float): 0-100
            - prob_down (float): 0-100
            - current_price (float)
            - features (dict): latest feature values
    """
    model = joblib.load(model_path)

    raw = yf.download(ticker, period="6mo", progress=False)
    df  = build_features(raw)

    latest     = df[FEATURES].iloc[-1:]
    prediction = int(model.predict(latest)[0])
    proba      = model.predict_proba(latest)[0]

    return {
        "ticker":        ticker,
        "prediction":    prediction,
        "signal":        "UP" if prediction == 1 else "DOWN",
        "confidence":    round(float(max(proba)) * 100, 2),
        "prob_up":       round(float(proba[1]) * 100, 2),
        "prob_down":     round(float(proba[0]) * 100, 2),
        "current_price": round(float(df['Close'].iloc[-1]), 2),
        "features":      latest.to_dict(orient='records')[0],
        "df":            df,
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict("AAPL")
    print(f"\nTicker:     {result['ticker']}")
    print(f"Signal:     {result['signal']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Price:      ${result['current_price']}")
