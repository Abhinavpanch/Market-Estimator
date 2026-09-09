"""
Loads the trained model and produces a live prediction for each ticker in
config.TICKERS, using the freshest available price data.

This is a ONE-SHOT script for now: run it, get a prediction, done. To make
this actually "live" for the dashboard, you'd wrap this logic in a small
API (e.g. FastAPI) with an endpoint like GET /predict/AAPL, and call it on
a schedule (e.g. every few minutes) from the front end -- rather than
re-running this whole script by hand.

Run this AFTER models/train_model.py has produced models/model.pkl.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from features.build_features import add_technical_features
from features.sentiment import fetch_company_news, aggregate_daily_sentiment

API_KEY = os.environ.get("FINNHUB_API_KEY", "")


def get_latest_sentiment(ticker: str):
    """
    Fetches the last 14 days of news for this ticker and returns
    (avg_sentiment, total_article_count). Falls back to (0.0, 0.0) if no
    API key is set or no news is found -- same fallback behavior as
    training, so live predictions don't error out just because sentiment
    is unavailable.
    """
    if not API_KEY:
        return 0.0, 0.0

    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
    articles = fetch_company_news(ticker, from_date, to_date, API_KEY)
    daily = aggregate_daily_sentiment(articles)

    if daily.empty:
        return 0.0, 0.0
    return float(daily["news_sentiment"].mean()), float(daily["news_volume"].sum())


def get_latest_features(ticker: str):
    # Pull enough recent history for rolling-window indicators to be valid
    # (e.g. the 50-day SMA needs 50+ days of data before it), plus a small buffer.
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Recent yfinance versions return a MultiIndex on columns (e.g.
    # ("Close", "AAPL")) even for a single ticker -- flatten it before
    # lowercasing, same fix as data/fetch_data.py.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = add_technical_features(df)
    df = df.dropna()

    if df.empty:
        raise ValueError(f"Not enough history to compute features for {ticker}")

    latest_row = df.iloc[[-1]].copy()
    latest_close = df["close"].iloc[-1]

    sentiment_score, news_volume = get_latest_sentiment(ticker)
    latest_row["news_sentiment"] = sentiment_score
    latest_row["news_volume"] = news_volume

    return latest_row, latest_close


def predict_ticker(ticker: str, bundle: dict):
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_row, latest_close = get_latest_features(ticker)
    prob_up = model.predict_proba(latest_row[feature_columns])[0, 1]
    direction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = prob_up if prob_up > 0.5 else 1 - prob_up

    return {
        "ticker": ticker,
        "latest_close": round(float(latest_close), 2),
        "direction": direction,
        "confidence": round(float(confidence) * 100, 1),
        "horizon_days": config.PREDICTION_HORIZON_DAYS,
    }


def main():
    if not config.MODEL_PATH.exists():
        print("No trained model found. Run models/train_model.py first.")
        return

    bundle = joblib.load(config.MODEL_PATH)

    print(f"Live predictions (horizon: {config.PREDICTION_HORIZON_DAYS} trading days)\n")
    for ticker in config.TICKERS:
        try:
            result = predict_ticker(ticker, bundle)
            print(
                f"{result['ticker']:6s}  "
                f"close=${result['latest_close']:<10} "
                f"prediction={result['direction']:5s} "
                f"confidence={result['confidence']}%"
            )
        except Exception as e:
            print(f"{ticker:6s}  ERROR: {e}")


if __name__ == "__main__":
    main()
