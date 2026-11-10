"""Evaluate one model on the latest reserved chronological period."""

import json
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from models.train_model import (
    FEATURE_COLUMNS,
    load_features,
    split_final_test_period,
)

_backtest_path = Path(__file__).resolve().parent / "backtest.py"
_spec = importlib.util.spec_from_file_location("backtest_core", _backtest_path)
_backtest_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backtest_core)

CONFIDENCE_THRESHOLD = _backtest_core.CONFIDENCE_THRESHOLD
TRANSACTION_COST_BPS = _backtest_core.TRANSACTION_COST_BPS
compute_buy_and_hold_equity_curve = _backtest_core.compute_buy_and_hold_equity_curve
compute_portfolio_equity_curve = _backtest_core.compute_portfolio_equity_curve
enforce_non_overlapping = _backtest_core.enforce_non_overlapping
risk_metrics = _backtest_core.risk_metrics
simulate_random_baseline = _backtest_core.simulate_random_baseline


def main():
    df = load_features()
    development, test, test_start = split_final_test_period(df)
    if development.empty or test.empty:
        raise ValueError("The reserved final period has no usable train or test rows.")

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, verbose=-1
    )
    model.fit(development[FEATURE_COLUMNS], development["label"])
    test = test.copy()
    test["pred_prob"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    signals = test[test["pred_prob"] > CONFIDENCE_THRESHOLD].copy()
    signals["future_return"] = signals["forward_return"]
    signals["net_return"] = (
        signals["future_return"] - TRANSACTION_COST_BPS / 10_000
    )
    trades = enforce_non_overlapping(signals)
    if trades.empty:
        end_date = test.index.max()
    else:
        end_date = max(
            test.index.max(),
            pd.to_datetime(trades["forward_exit_date"]).max(),
        )

    portfolio_equity = compute_portfolio_equity_curve(
        df, trades, config.TICKERS, test.index.min(), end_date
    )
    benchmark_equity = compute_buy_and_hold_equity_curve(
        df, config.TICKERS, test.index.min(), end_date
    )
    portfolio_multiple = float(portfolio_equity.iloc[-1])
    benchmark_multiple = float(benchmark_equity.iloc[-1])
    random_multiple = simulate_random_baseline(test, trades, len(config.TICKERS))
    labels = test["label"]
    probabilities = test["pred_prob"]

    summary = {
        "testStart": test_start.strftime("%Y-%m-%d"),
        "testEnd": end_date.strftime("%Y-%m-%d"),
        "developmentRows": int(len(development)),
        "testRows": int(len(test)),
        "trades": int(len(trades)),
        "accuracy": round(float(accuracy_score(labels, probabilities >= 0.5)), 4),
        "auc": round(float(roc_auc_score(labels, probabilities)), 4)
        if labels.nunique() > 1 else None,
        "brierScore": round(float(brier_score_loss(labels, probabilities)), 4),
        "modelReturn": round((portfolio_multiple - 1) * 100, 2),
        "benchmarkReturn": round((benchmark_multiple - 1) * 100, 2),
        "randomReturn": round((random_multiple - 1) * 100, 2),
        "risk": risk_metrics(portfolio_equity, trades),
        "assumedRoundTripCostBps": TRANSACTION_COST_BPS,
        "historicallyPreviouslyInspected": True,
        "warning": (
            "This historical period was examined in earlier runs, so it is "
            "not an untouched test. Do not tune from these results; prospective "
            "evaluation after this run is needed for an independent test."
        ),
    }

    out_path = config.RESULTS_DIR / "final_period.json"
    out_path.write_text(json.dumps({
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
    }, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved reserved-period evaluation to {out_path}")


if __name__ == "__main__":
    main()
