"""
Trains on earlier dates from config.TICKERS and evaluates later periods
on config.HOLDOUT_TICKERS -- stocks never included in the training universe.

For each out-of-time period, the model trains only on earlier dates from
the training universe, then evaluates on the same period for the separate
holdout stocks. This checks both chronological and ticker separation.

Run this after:
  1. data/fetch_holdout_data.py
  2. features/build_holdout_features.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import (
    FEATURE_COLUMNS,
    calibration_summary,
    load_features,
    walk_forward_splits,
)

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
compute_buy_and_hold_equity_curve = _backtest_core.compute_buy_and_hold_equity_curve
compute_portfolio_equity_curve = _backtest_core.compute_portfolio_equity_curve
risk_metrics = _backtest_core.risk_metrics
print_per_ticker_breakdown = _backtest_core.print_per_ticker_breakdown
simulate_random_baseline = _backtest_core.simulate_random_baseline
CONFIDENCE_THRESHOLD = _backtest_core.CONFIDENCE_THRESHOLD
TRANSACTION_COST_BPS = config.ROUND_TRIP_COST_BPS


def main():
    if not config.HOLDOUT_FEATURES_PATH.exists():
        print("No holdout features found. Run:")
        print("  python data/fetch_holdout_data.py")
        print("  python features/build_holdout_features.py")
        return

    train_features = load_features()
    holdout = pd.read_csv(
        config.HOLDOUT_FEATURES_PATH, index_col="date", parse_dates=True
    ).sort_index()
    if set(train_features["ticker"]) & set(holdout["ticker"]):
        raise ValueError("Training and holdout ticker sets overlap.")

    predictions = []
    period_metrics = []
    for fold_i, (train_df, test_df) in enumerate(
        walk_forward_splits(train_features, config.N_WALKFORWARD_SPLITS), start=1
    ):
        start, end = test_df.index.min(), test_df.index.max()
        fold_test = holdout[(holdout.index >= start) & (holdout.index <= end)].copy()
        if fold_test.empty:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, verbose=-1
        )
        model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        fold_test["pred_prob"] = model.predict_proba(
            fold_test[FEATURE_COLUMNS]
        )[:, 1]
        fold_test["fold"] = fold_i
        predictions.append(fold_test)
        period_metrics.append({
            "fold": fold_i,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "rows": int(len(fold_test)),
            "brierScore": round(
                float(brier_score_loss(fold_test["label"], fold_test["pred_prob"])), 4
            ),
        })

    if not predictions:
        raise ValueError("No matching holdout rows in the out-of-time test periods.")
    oos = pd.concat(predictions).sort_index()
    signals = oos[oos["pred_prob"] > CONFIDENCE_THRESHOLD].copy()
    cost = TRANSACTION_COST_BPS / 10_000
    signals["future_return"] = signals["forward_return"]
    signals["net_return"] = signals["future_return"] - cost

    trades = enforce_non_overlapping(signals)
    n_tickers = len(config.HOLDOUT_TICKERS)
    evaluation_start = oos.index.min()
    evaluation_end = max(
        oos.index.max(),
        pd.to_datetime(trades["forward_exit_date"]).max()
        if not trades.empty else oos.index.max(),
    )
    portfolio_equity = compute_portfolio_equity_curve(
        holdout, trades, config.HOLDOUT_TICKERS,
        evaluation_start, evaluation_end,
    )
    benchmark_equity = compute_buy_and_hold_equity_curve(
        holdout, config.HOLDOUT_TICKERS,
        evaluation_start, evaluation_end,
    )
    portfolio_multiple = float(portfolio_equity.iloc[-1])
    benchmark_multiple = float(benchmark_equity.iloc[-1])
    portfolio_risk = risk_metrics(portfolio_equity, trades)

    win_rate = (trades["net_return"] > 0).mean() if not trades.empty else 0.0
    random_multiple = simulate_random_baseline(oos, trades, n_tickers)

    for period in period_metrics:
        fold_predictions = oos[oos["fold"] == period["fold"]]
        fold_trades = enforce_non_overlapping(
            fold_predictions[
                fold_predictions["pred_prob"] > CONFIDENCE_THRESHOLD
            ].copy()
        )
        fold_trades["future_return"] = fold_trades["forward_return"]
        fold_trades["net_return"] = (
            fold_trades["future_return"] - TRANSACTION_COST_BPS / 10_000
        )
        period["trades"] = int(len(fold_trades))
        period_start = fold_predictions.index.min()
        period_end = max(
            fold_predictions.index.max(),
            pd.to_datetime(fold_trades["forward_exit_date"]).max()
            if not fold_trades.empty else fold_predictions.index.max(),
        )
        fold_equity = compute_portfolio_equity_curve(
            holdout, fold_trades, config.HOLDOUT_TICKERS,
            period_start, period_end,
        )
        fold_benchmark = compute_buy_and_hold_equity_curve(
            holdout, config.HOLDOUT_TICKERS, period_start, period_end
        )
        period["modelReturn"] = round(
            (float(fold_equity.iloc[-1]) - 1) * 100, 2
        )
        period["benchmarkReturn"] = round(
            (float(fold_benchmark.iloc[-1]) - 1) * 100, 2
        )

    print(f"Unseen holdout tickers: {config.HOLDOUT_TICKERS}")
    print("Each period's model trains only on earlier dates from training tickers.")
    for period in period_metrics:
        print(
            f"Fold {period['fold']}: {period['start']} to {period['end']} "
            f"rows={period['rows']} trades={period['trades']} "
            f"model={period['modelReturn']:.2f}% "
            f"buy-and-hold={period['benchmarkReturn']:.2f}% "
            f"Brier={period['brierScore']:.3f}"
        )
    print(f"Trades taken:                  {len(trades)}")
    print(f"Win rate (after costs):        {win_rate:.1%}")
    print(f"Compounded portfolio return:    {(portfolio_multiple - 1):.2%}")
    print(
        "Risk (mark-to-market):          "
        f"max drawdown {portfolio_risk['maxDrawdown']}%, "
        f"annualized vol {portfolio_risk['annualizedVolatility']}%, "
        f"Sharpe {portfolio_risk['annualizedSharpe']}"
    )
    print(f"Buy-and-hold benchmark:         {(benchmark_multiple - 1):.2%}")
    print(f"Random-entry baseline:          {(random_multiple - 1):.2%}")
    if portfolio_multiple < random_multiple:
        print(
            f"Model underperformed random timing by "
            f"{random_multiple - portfolio_multiple:.2%}."
        )

    edge = (portfolio_multiple - 1) - (benchmark_multiple - 1)
    if edge > 0:
        print(f"\nModel beat buy-and-hold on UNSEEN stocks by: {edge:.2%}")
        print("This is a real positive sign -- the edge isn't limited to the")
        print("5 training tickers.")
    else:
        print(f"\nModel UNDERPERFORMED buy-and-hold on unseen stocks by: {abs(edge):.2%}")
        print("This suggests the earlier result on the training tickers was")
        print("likely specific to those 5 stocks' history, not a general pattern.")

    print(f"Assumed round-trip fees + slippage: {TRANSACTION_COST_BPS} bps")
    if not trades.empty:
        print_per_ticker_breakdown(trades, n_tickers)
    save_holdout_json(
        trades, n_tickers, win_rate, portfolio_multiple, benchmark_multiple,
        random_multiple, oos, period_metrics, portfolio_risk,
    )


def save_holdout_json(
    trades, n_tickers, win_rate, portfolio_multiple, benchmark_multiple,
    random_multiple, oos, period_metrics, portfolio_risk,
):
    """
    Writes results/holdout_backtest.json -- read by the dashboard server.
    Overwritten every time this script runs.
    """
    output = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "trades": int(len(trades)),
            "winRate": round(float(win_rate) * 100, 1),
            "modelReturn": round((portfolio_multiple - 1) * 100, 2),
            "benchmarkReturn": round((benchmark_multiple - 1) * 100, 2),
            "randomReturn": round((random_multiple - 1) * 100, 2),
            "risk": portfolio_risk,
        },
        "tickers": compute_per_ticker_stats(trades, n_tickers),
        "assumedRoundTripCostBps": TRANSACTION_COST_BPS,
        "calibration": calibration_summary(oos["label"], oos["pred_prob"]),
        "periodMetrics": period_metrics,
        "costSensitivity": _backtest_core.compute_cost_sensitivity(trades, n_tickers),
    }

    out_path = config.RESULTS_DIR / "holdout_backtest.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path} (dashboard will pick this up automatically)")


if __name__ == "__main__":
    main()
