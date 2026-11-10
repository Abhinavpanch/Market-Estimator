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
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from features.build_features import add_technical_features
from features.sentiment import fetch_company_news, aggregate_daily_sentiment


def forecast_target_date(today: date | None = None) -> str:
    today = today or date.today()
    return (today + timedelta(days=config.PREDICTION_CALENDAR_DAYS)).isoformat()


def get_latest_sentiment(ticker: str, sentiment_date):
    """
    Fetches sentiment for the prior market session, matching training's
    one-session lag. Returns zero features when no key/news is available.
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key or sentiment_date is None:
        return 0.0, 0.0

    date = pd.Timestamp(sentiment_date).strftime("%Y-%m-%d")
    articles = fetch_company_news(ticker, date, date, api_key)
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
    df = df.sort_index()

    # Recent yfinance versions return a MultiIndex on columns (e.g.
    # ("Close", "AAPL")) even for a single ticker -- flatten it before
    # lowercasing, same fix as data/fetch_data.py.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    chart_history = df.tail(90).copy()
    df = add_technical_features(df)
    df = df.dropna()

    if df.empty:
        raise ValueError(f"Not enough history to compute features for {ticker}")

    latest_row = df.iloc[[-1]].copy()
    latest_close = df["close"].iloc[-1]

    prior_session = df.index[-2] if len(df.index) > 1 else None
    sentiment_score, news_volume = get_latest_sentiment(ticker, prior_session)
    latest_row["news_sentiment"] = sentiment_score
    latest_row["news_volume"] = news_volume

    chart_history.index = pd.to_datetime(chart_history.index)
    history = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for date, row in chart_history.iterrows()
    ]
    return latest_row, latest_close, history


def predict_ticker(ticker: str, bundle: dict):
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_row, latest_close, history = get_latest_features(ticker)
    raw_prob_up = model.predict_proba(latest_row[feature_columns])[0, 1]
    calibrator = bundle.get("probability_calibrator")
    prob_up = (
        float(calibrator.predict_proba([[raw_prob_up]])[0, 1])
        if calibrator is not None else float(raw_prob_up)
    )
    direction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = prob_up if prob_up > 0.5 else 1 - prob_up
    sentiment_articles = (
        float(latest_row["news_volume"].iloc[0]) > 0
        if "news_volume" in latest_row else False
    )
    feature_importances = getattr(model, "feature_importances_", [])
    sentiment_importance = sum(
        float(importance)
        for feature, importance in zip(feature_columns, feature_importances)
        if feature in ("news_sentiment", "news_volume")
    )
    daily_volatility = float(latest_row["volatility_10"].iloc[0])
    illustrative_change = (
        latest_close * daily_volatility * config.PREDICTION_HORIZON_DAYS ** 0.5
    )
    illustrative_close = latest_close + (
        illustrative_change if direction == "UP" else -illustrative_change
    )

    return {
        "ticker": ticker,
        "latest_close": round(float(latest_close), 2),
        "direction": direction,
        "confidence": round(float(confidence) * 100, 1),
        "probability_up": round(float(prob_up) * 100, 1),
        "probability_calibrated": calibrator is not None,
        "sentiment_key_configured": bool(os.environ.get("FINNHUB_API_KEY")),
        "sentiment_articles": sentiment_articles,
        "sentiment_score": round(float(latest_row["news_sentiment"].iloc[0]), 3),
        "sentiment_article_count": int(latest_row["news_volume"].iloc[0]),
        "sentiment_model_uses_feature": sentiment_importance > 0,
        "sentiment_model_feature_importance": sentiment_importance,
        "price_as_of": history[-1]["date"] if history else None,
        "horizon_days": config.PREDICTION_HORIZON_DAYS,
        "horizon_calendar_days": config.PREDICTION_CALENDAR_DAYS,
        "forecast_date": forecast_target_date(),
        "history": history,
        "illustrative_close": round(float(illustrative_close), 2),
    }


def main():
    if not config.MODEL_PATH.exists():
        print("No trained model found. Run models/train_model.py first.")
        return

    bundle = joblib.load(config.MODEL_PATH)

    print(
        f"Live predictions (about {config.PREDICTION_CALENDAR_DAYS} calendar days, "
        f"{config.PREDICTION_HORIZON_DAYS} trading sessions)\n"
    )
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
