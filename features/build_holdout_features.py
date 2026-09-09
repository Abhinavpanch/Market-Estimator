"""
Builds features for config.HOLDOUT_TICKERS using the exact same feature
logic as the main pipeline, but saves to a SEPARATE file
(holdout_features.csv) so this data never accidentally gets mixed into
model training.

Run this after data/fetch_holdout_data.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from features.build_features import build_features_for_ticker


def main():
    all_frames = []
    for ticker in config.HOLDOUT_TICKERS:
        print(f"Building holdout features for {ticker}...")
        try:
            df = build_features_for_ticker(ticker)
        except FileNotFoundError:
            print(f"  no raw data found for {ticker} -- run data/fetch_holdout_data.py first.")
            continue
        all_frames.append(df)
        coverage = (df["news_sentiment"] != 0).mean()
        print(f"  {len(df)} labeled rows, {coverage:.1%} have real sentiment data")

    if not all_frames:
        print("No holdout data found. Run data/fetch_holdout_data.py first.")
        return

    combined = pd.concat(all_frames)
    combined.to_csv(config.HOLDOUT_FEATURES_PATH)
    print(f"\nSaved holdout features for {len(all_frames)} tickers "
          f"({len(combined)} rows) to {config.HOLDOUT_FEATURES_PATH}")
    print("Next step: python backtest/test_holdout.py")


if __name__ == "__main__":
    main()
