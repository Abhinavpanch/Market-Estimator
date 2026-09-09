"""
Turns raw OHLCV data into model-ready features + labels.

This is the highest-leverage file in the whole project. The model can only
ever be as good as the signal in these features -- start here when you want
to improve accuracy, not by swapping the model architecture.

Label: 1 if the close price is higher PREDICTION_HORIZON_DAYS ahead, else 0.
This makes it a binary classification problem (direction), not exact price.
"""

import sys
from pathlib import Path

import pandas as pd
import ta  # technical analysis indicator library

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Momentum / trend ---
    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # --- Volatility ---
    df["volatility_10"] = close.pct_change().rolling(10).std()
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close

    # --- Volume ---
    df["volume_change_5"] = volume.pct_change(5)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # --- Recent return features ---
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    return df


def merge_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Left-joins daily news sentiment onto the feature dataframe by date.
    If no sentiment file exists for this ticker (fetch_news_sentiment.py
    was never run, or no Finnhub API key was set), fills with 0 so the
    pipeline still works without this optional signal -- it just won't
    carry any information in that case.
    """
    sentiment_path = config.NEWS_SENTIMENT_DIR / f"{ticker}.csv"
    if sentiment_path.exists():
        sentiment_df = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        df = df.join(sentiment_df, how="left")
    else:
        df["news_sentiment"] = pd.NA
        df["news_volume"] = pd.NA

    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
    df["news_volume"] = df["news_volume"].fillna(0.0)
    return df


def add_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Label = 1 if price is higher `horizon` trading days in the future.

    IMPORTANT: this uses .shift(-horizon), which looks FORWARD in time.
    That's correct for building a label, but it means the last `horizon`
    rows of each ticker's data have no valid label (nothing to compare to)
    and must be dropped before training -- handled in the caller.
    """
    future_close = df["close"].shift(-horizon)
    df["label"] = (future_close > df["close"]).astype(int)
    return df


def build_features_for_ticker(ticker: str) -> pd.DataFrame:
    raw_path = config.RAW_DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    df = add_technical_features(df)
    df = merge_sentiment(df, ticker)
    df = add_label(df, config.PREDICTION_HORIZON_DAYS)
    df["ticker"] = ticker

    # Drop rows with NaNs from rolling windows AND the un-labeled tail rows
    df = df.dropna()
    return df


def main():
    all_frames = []
    total_sentiment_coverage = []

    for ticker in config.TICKERS:
        print(f"Building features for {ticker}...")
        try:
            df = build_features_for_ticker(ticker)
        except FileNotFoundError:
            print(f"  no raw data found for {ticker} -- run data/fetch_data.py first.")
            continue
        all_frames.append(df)
        coverage = (df["news_sentiment"] != 0).mean()
        total_sentiment_coverage.append(coverage)
        print(f"  {len(df)} labeled rows, {coverage:.1%} have real sentiment data")

    combined = pd.concat(all_frames)
    combined.to_csv(config.FEATURES_PATH)

    if total_sentiment_coverage:
        avg_coverage = sum(total_sentiment_coverage) / len(total_sentiment_coverage)
        print(f"\nAverage sentiment coverage across all tickers: {avg_coverage:.1%} of rows")
        if avg_coverage < 0.10:
            print(
                "This is low -- expected given free-tier historical news limits. "
                "The sentiment feature likely won't move training/backtest results "
                "much, since the model sees mostly zeros. It may still help live "
                "predictions, where only recent news matters."
            )

    print(f"\nSaved combined features for {len(all_frames)} tickers "
          f"({len(combined)} rows) to {config.FEATURES_PATH}")
    print("Next step: python models/train_model.py")


if __name__ == "__main__":
    main()
