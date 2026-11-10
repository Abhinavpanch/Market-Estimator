"""
Re-runs the walk-forward backtest at several CONFIDENCE_THRESHOLD values
and prints them side by side. A real edge should degrade gracefully as
the threshold changes; a result that only appears at one exact threshold
value (e.g. 0.55 works but 0.52 and 0.6 both fail) is a strong sign of a
lucky/overfit result rather than a real pattern.

Run this any time after models/train_model.py has produced features.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import walk_forward_splits, load_features

# Load backtest.py directly by file path rather than "from backtest.backtest
# import ..." -- because this folder is named "backtest" AND contains a
# file called "backtest.py", Python's normal package import gets confused
# about which one is which when running a script from inside this same
# folder. Loading by explicit path sidesteps that entirely.
import importlib.util
_backtest_path = Path(__file__).resolve().parent / "backtest.py"
_spec = importlib.util.spec_from_file_location("backtest_core", _backtest_path)
_backtest_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backtest_core)

simulate_fold = _backtest_core.simulate_fold
enforce_non_overlapping = _backtest_core.enforce_non_overlapping
compute_portfolio_equity_curve = _backtest_core.compute_portfolio_equity_curve
compute_buy_and_hold_equity_curve = _backtest_core.compute_buy_and_hold_equity_curve

THRESHOLDS_TO_TEST = [0.52, 0.55, 0.58, 0.60, 0.65]


def run_for_threshold(df: pd.DataFrame, oof_predictions: pd.DataFrame,
                      threshold: float):
    trades = enforce_non_overlapping(
        oof_predictions[oof_predictions["pred_prob"] > threshold]
    )
    if trades.empty:
        return None

    start_date = oof_predictions.index.min()
    end_date = max(
        oof_predictions.index.max(),
        pd.to_datetime(trades["forward_exit_date"]).max(),
    )
    portfolio_equity = compute_portfolio_equity_curve(
        df, trades, config.TICKERS, start_date, end_date
    )
    benchmark_equity = compute_buy_and_hold_equity_curve(
        df, config.TICKERS, start_date, end_date
    )

    return {
        "threshold": threshold,
        "trades": len(trades),
        "win_rate": (trades["net_return"] > 0).mean(),
        "portfolio_return": float(portfolio_equity.iloc[-1]) - 1,
        "benchmark_return": float(benchmark_equity.iloc[-1]) - 1,
    }


def main():
    df = load_features()
    predictions = []
    for fold_i, (train_df, test_df) in enumerate(
        walk_forward_splits(df, config.N_WALKFORWARD_SPLITS), start=1
    ):
        fold_predictions = simulate_fold(train_df, test_df)
        fold_predictions["fold"] = fold_i
        predictions.append(fold_predictions)
    if not predictions:
        raise ValueError("No walk-forward predictions were produced.")
    oof_predictions = pd.concat(predictions)
    print("Sweeping thresholds over the same out-of-sample predictions.\n")

    print(f"{'THRESHOLD':>10s}{'TRADES':>9s}{'WIN RATE':>10s}{'MODEL RET':>12s}{'BUY&HOLD':>11s}{'BEAT?':>8s}")
    for threshold in THRESHOLDS_TO_TEST:
        result = run_for_threshold(df, oof_predictions, threshold)
        if result is None:
            print(f"{threshold:>10.2f}   no trades taken at this threshold")
            continue
        beat = "YES" if result["portfolio_return"] > result["benchmark_return"] else "no"
        print(
            f"{result['threshold']:>10.2f}"
            f"{result['trades']:>9d}"
            f"{result['win_rate']:>9.1%} "
            f"{result['portfolio_return']:>11.1%}"
            f"{result['benchmark_return']:>11.1%}"
            f"{beat:>8s}"
        )

    print(
        "\nLook for a consistent pattern across thresholds, not just one good "
        "number. If only one threshold beats buy-and-hold and the neighbors "
        "don't, treat the original result as likely noise."
    )


if __name__ == "__main__":
    main()
