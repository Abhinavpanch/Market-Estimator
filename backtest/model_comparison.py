"""Compare a few predeclared model/feature baselines on development folds only."""

import json
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from models.train_model import (
    FEATURE_COLUMNS,
    final_test_start,
    load_features,
    walk_forward_splits,
)

_backtest_path = Path(__file__).resolve().parent / "backtest.py"
_spec = importlib.util.spec_from_file_location("backtest_core", _backtest_path)
_backtest_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backtest_core)

CONFIDENCE_THRESHOLD = _backtest_core.CONFIDENCE_THRESHOLD
TRANSACTION_COST_BPS = _backtest_core.TRANSACTION_COST_BPS
compute_portfolio_equity_curve = _backtest_core.compute_portfolio_equity_curve
enforce_non_overlapping = _backtest_core.enforce_non_overlapping
risk_metrics = _backtest_core.risk_metrics

TECHNICAL_COLUMNS = [c for c in FEATURE_COLUMNS if not c.startswith("news_")]
VARIANTS = {
    "lightgbm_all_features": (FEATURE_COLUMNS, "lightgbm"),
    "logistic_all_features": (FEATURE_COLUMNS, "logistic"),
    "lightgbm_technical_only": (TECHNICAL_COLUMNS, "lightgbm"),
}


def main():
    df = load_features()
    predictions = {name: [] for name in VARIANTS}

    for fold, (train, test) in enumerate(
        walk_forward_splits(df, config.N_WALKFORWARD_SPLITS), start=1
    ):
        for name, (columns, model_type) in VARIANTS.items():
            if model_type == "logistic":
                valid_train = np.isfinite(train[columns]).all(axis=1)
                valid_test = np.isfinite(test[columns]).all(axis=1)
                train_rows, test_rows = train.loc[valid_train], test.loc[valid_test]
            else:
                train_rows, test_rows = train, test
            if train_rows["label"].nunique() < 2 or test_rows.empty:
                continue

            if model_type == "logistic":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, random_state=42),
                )
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    verbose=-1,
                )
            model.fit(train_rows[columns], train_rows["label"])
            fold_predictions = test_rows[
                ["ticker", "label", "forward_return", "forward_exit_date"]
            ].copy()
            fold_predictions["pred_prob"] = model.predict_proba(
                test_rows[columns]
            )[:, 1]
            fold_predictions["fold"] = fold
            predictions[name].append(fold_predictions)

    results = {}
    evaluation_end = df.index[df.index < final_test_start(df)].max()
    for name, folds in predictions.items():
        if not folds:
            raise ValueError(f"No valid walk-forward predictions for {name}.")
        oof = pd.concat(folds).sort_index()
        signals = oof[oof["pred_prob"] > CONFIDENCE_THRESHOLD].copy()
        signals["future_return"] = signals["forward_return"]
        signals["net_return"] = (
            signals["future_return"] - TRANSACTION_COST_BPS / 10_000
        )
        trades = enforce_non_overlapping(signals)
        end_date = max(
            evaluation_end,
            pd.to_datetime(trades["forward_exit_date"]).max()
            if not trades.empty else evaluation_end,
        )
        equity = compute_portfolio_equity_curve(
            df, trades, config.TICKERS, oof.index.min(), end_date
        )
        labels, probabilities = oof["label"], oof["pred_prob"]
        results[name] = {
            "rows": int(len(oof)),
            "trades": int(len(trades)),
            "accuracy": round(
                float(accuracy_score(labels, probabilities >= 0.5)), 4
            ),
            "auc": round(float(roc_auc_score(labels, probabilities)), 4)
            if labels.nunique() > 1 else None,
            "brierScore": round(float(brier_score_loss(labels, probabilities)), 4),
            "modelReturn": round((float(equity.iloc[-1]) - 1) * 100, 2),
            "risk": risk_metrics(equity, trades),
        }

    output = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "threshold": CONFIDENCE_THRESHOLD,
        "assumedRoundTripCostBps": TRANSACTION_COST_BPS,
        "reservedFinalPeriodExcluded": True,
        "warning": (
            "These development-fold comparisons are for baseline diagnosis, "
            "not proof of a tradable edge. Do not choose a variant using the "
            "reserved final period."
        ),
        "variants": results,
    }
    out_path = config.RESULTS_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"\nSaved development-only comparison to {out_path}")


if __name__ == "__main__":
    main()
