"""
One-command pipeline runner: fetch data -> (optional) fetch sentiment ->
build features -> train model -> backtest -> (optional) holdout test.

Run this instead of calling each script by hand every time you want to
retrain on fresh data -- e.g. weekly, or after editing config.TICKERS.

Usage:
    python retrain_all.py                        # price data + train + backtest
    python retrain_all.py --sentiment            # also refresh news sentiment
    python retrain_all.py --holdout              # also re-validate on holdout tickers
    python retrain_all.py --sentiment --holdout  # everything

Run this from the project root (same place you'd run any other script
from, e.g. "py retrain_all.py").
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

_step_num = 0


def step(name: str, fn) -> None:
    global _step_num
    _step_num += 1
    print(f"\n{'=' * 60}\nStep {_step_num}: {name}\n{'=' * 60}")
    start = time.time()
    fn()
    print(f"\n({name} took {time.time() - start:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the AI Stock Estimator pipeline end to end."
    )
    parser.add_argument("--sentiment", action="store_true",
                         help="Also fetch/refresh news sentiment (needs FINNHUB_API_KEY set)")
    parser.add_argument("--holdout", action="store_true",
                         help="Also rebuild holdout features and re-run the holdout test")
    args = parser.parse_args()

    import backtest.backtest as backtest
    import data.fetch_data as fetch_data
    import features.build_features as build_features
    import models.train_model as train_model

    step("Fetching price data", fetch_data.main)

    if args.sentiment:
        import data.fetch_news_sentiment as fetch_sentiment
        step("Fetching news sentiment", fetch_sentiment.main)

    if args.holdout:
        import data.fetch_holdout_data as fetch_holdout
        step("Fetching holdout price data", fetch_holdout.main)

    step("Building features", build_features.main)

    if args.holdout:
        import features.build_holdout_features as build_holdout_features
        step("Building holdout features", build_holdout_features.main)

    step("Training model", train_model.main)
    step("Running backtest", backtest.main)

    if args.holdout:
        import backtest.test_holdout as test_holdout
        step("Running holdout test", test_holdout.main)

    print(
        "\nAll done. If dashboard/server.py is running, restart it "
        "(Ctrl+C, then re-run) so it picks up the new model and results."
    )


if __name__ == "__main__":
    main()
