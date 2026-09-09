"""
Loads the model that was trained ONLY on config.TICKERS (AAPL, MSFT,
GOOGL, AMZN, NVDA) and tests it on config.HOLDOUT_TICKERS -- stocks it has
never seen in training or in any walk-forward fold.

This is the single most important check: if the model's edge disappears
here, it learned something specific to the 5 training stocks (their
particular history, their particular bull run) rather than a pattern that
generalizes to other stocks. That's the difference between a real
(if modest) trading signal and a model that memorized its own training
data's story.

Run this after:
  1. data/fetch_holdout_data.py
  2. features/build_holdout_features.py
  3. models/train_model.py (the model must already be trained)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import FEATURE_COLUMNS

# Loaded by explicit file path rather than "from backtest.backtest import
# ..." -- see the comment in threshold_sweep.py for why the normal package
# import breaks here (folder "backtest" contains a file "backtest.py").
import importlib.util
_backtest_path = Path(__file__).resolve().parent / "backtest.py"
_spec = importlib.util.spec_from_file_location("backtest_core", _backtest_path)
_backtest_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backtest_core)

compute_per_ticker_stats = _backtest_core.compute_per_ticker_stats
enforce_non_overlapping = _backtest_core.enforce_non_overlapping
compute_equity_curve = _backtest_core.compute_equity_curve
compute_buy_and_hold_benchmark = _backtest_core.compute_buy_and_hold_benchmark
print_per_ticker_breakdown = _backtest_core.print_per_ticker_breakdown
CONFIDENCE_THRESHOLD = _backtest_core.CONFIDENCE_THRESHOLD
TRANSACTION_COST_BPS = _backtest_core.TRANSACTION_COST_BPS


def main():
    if not config.MODEL_PATH.exists():
        print("No trained model found. Run models/train_model.py first.")
        return
    if not config.HOLDOUT_FEATURES_PATH.exists():
        print("No holdout features found. Run:")
        print("  python data/fetch_holdout_data.py")
        print("  python features/build_holdout_features.py")
        return

    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]

    df = pd.read_csv(config.HOLDOUT_FEATURES_PATH, index_col="date", parse_dates=True)
    df = df.sort_index()

    df["pred_prob"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["future_return"] = df.groupby("ticker")["close"].pct_change(
        config.PREDICTION_HORIZON_DAYS
    ).shift(-config.PREDICTION_HORIZON_DAYS)

    signals = df[df["pred_prob"] > CONFIDENCE_THRESHOLD].dropna(subset=["future_return"])
    if signals.empty:
        print("The model never fired a confident signal on any holdout ticker.")
        print("That itself is informative -- it suggests the model's confidence")
        print("is tied to something specific about the training tickers.")
        return

    cost = TRANSACTION_COST_BPS / 10_000
    signals = signals.copy()
    signals["net_return"] = signals["future_return"] - cost

    trades = enforce_non_overlapping(signals)
    n_tickers = trades["ticker"].nunique()
    portfolio_multiple = compute_equity_curve(trades, n_tickers)

    benchmark_multiple = compute_buy_and_hold_benchmark(
        df, config.HOLDOUT_TICKERS, [df.index.min()]
    )

    win_rate = (trades["net_return"] > 0).mean()

    print(f"Holdout tickers (never trained on): {config.HOLDOUT_TICKERS}\n")
    print(f"Trades taken:                  {len(trades)}")
    print(f"Win rate (after costs):        {win_rate:.1%}")
    print(f"Compounded portfolio return:    {(portfolio_multiple - 1):.2%}")
    print(f"Buy-and-hold benchmark:         {(benchmark_multiple - 1):.2%}")

    edge = (portfolio_multiple - 1) - (benchmark_multiple - 1)
    if edge > 0:
        print(f"\nModel beat buy-and-hold on UNSEEN stocks by: {edge:.2%}")
        print("This is a real positive sign -- the edge isn't limited to the")
        print("5 training tickers.")
    else:
        print(f"\nModel UNDERPERFORMED buy-and-hold on unseen stocks by: {abs(edge):.2%}")
        print("This suggests the earlier result on the training tickers was")
        print("likely specific to those 5 stocks' history, not a general pattern.")

    print_per_ticker_breakdown(trades, n_tickers)
    save_holdout_json(trades, n_tickers, win_rate, portfolio_multiple, benchmark_multiple)


def save_holdout_json(trades, n_tickers, win_rate, portfolio_multiple, benchmark_multiple):
    """
    Writes results/holdout_backtest.json -- read by the dashboard server.
    Overwritten every time this script runs.
    """
    import json
    from datetime import datetime

    output = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "trades": int(len(trades)),
            "winRate": round(float(win_rate) * 100, 1),
            "modelReturn": round((portfolio_multiple - 1) * 100, 2),
            "benchmarkReturn": round((benchmark_multiple - 1) * 100, 2),
        },
        "tickers": compute_per_ticker_stats(trades, n_tickers),
    }

    out_path = config.RESULTS_DIR / "holdout_backtest.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path} (dashboard will pick this up automatically)")


if __name__ == "__main__":
    main()
