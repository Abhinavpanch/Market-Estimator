"""
Trains a gradient-boosted classifier to predict price direction, using
walk-forward (time-ordered) validation instead of random train/test splits.

WHY WALK-FORWARD MATTERS:
Randomly shuffling time-series data into train/test lets the model "see"
data from after the test period during training (via overlapping rolling
features), which leaks future information and makes accuracy look far
better than it would in real, live use. Walk-forward splits always train on the past and test on later data.
Training rows are purged so their forward outcome is already known by
the time the test period begins.

This script:
  1. Loads features.csv (from features/build_features.py)
  2. Sorts by date
  3. Runs N walk-forward folds, training on the past and testing on the future
  4. Reports honest, out-of-sample accuracy for each development fold
  5. Refits on development data and saves the final model for live use
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

FEATURE_COLUMNS = [
    "sma_ratio", "rsi_14", "macd", "macd_signal",
    "volatility_10", "bb_width",
    "volume_change_5", "obv",
    "return_1d", "return_5d", "return_10d",
    "news_sentiment", "news_volume",
]


def final_test_start(df: pd.DataFrame):
    if not 0 < config.FINAL_TEST_FRACTION < 1:
        raise ValueError("FINAL_TEST_FRACTION must be between 0 and 1.")
    unique_dates = df.index.unique().sort_values()
    if len(unique_dates) < 3:
        raise ValueError("Not enough unique dates to reserve a final test period.")
    test_start_pos = int(len(unique_dates) * (1 - config.FINAL_TEST_FRACTION))
    return unique_dates[min(max(test_start_pos, 1), len(unique_dates) - 1)]


def split_final_test_period(df: pd.DataFrame):
    """Split by date and purge training rows whose outcomes reach the test."""
    test_start = final_test_start(df)
    train = df[df.index < test_start]
    train = train[pd.to_datetime(train["forward_exit_date"]) < test_start]
    test = df[
        (df.index >= test_start)
        & (pd.to_datetime(df["forward_exit_date"]) <= df.index.max())
    ]
    return train, test, test_start


def load_features() -> pd.DataFrame:
    df = pd.read_csv(config.FEATURES_PATH, index_col="date", parse_dates=True)
    df = df.sort_index()
    return df


def walk_forward_splits(
    df: pd.DataFrame, n_splits: int, reserve_final: bool = True
):
    """
    Yields (train_df, test_df) tuples, each test fold strictly AFTER its
    corresponding train fold in time. This mirrors how the model will
    actually be used: trained on the past, evaluated on data it hasn't
    seen yet.
    """
    test_start = final_test_start(df) if reserve_final else None
    development = df[df.index < test_start] if reserve_final else df
    unique_dates = development.index.unique().sort_values()
    fold_size = len(unique_dates) // (n_splits + 1)
    if fold_size == 0:
        raise ValueError("Not enough unique dates for the requested walk-forward folds.")

    for i in range(1, n_splits + 1):
        train_end_pos = fold_size * i
        train_end = unique_dates[train_end_pos]
        test_end = unique_dates[min(fold_size * (i + 1) - 1, len(unique_dates) - 1)]
        train_df = df[df.index <= train_end]
        train_df = train_df[
            pd.to_datetime(train_df["forward_exit_date"]) <= train_end
        ]
        test_df = df[(df.index > train_end) & (df.index <= test_end)]
        if reserve_final:
            test_df = test_df[
                pd.to_datetime(test_df["forward_exit_date"]) < test_start
            ]
        if train_df.empty or test_df.empty:
            continue
        yield train_df, test_df


def calibration_summary(labels, probabilities):
    """Return Brier score and reliability bins for probability estimates."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels):
        return {"brierScore": None, "bins": []}

    bins = []
    edges = np.linspace(0, 1, 6)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (
            probabilities <= high if high == 1 else probabilities < high
        )
        if mask.any():
            bins.append({
                "range": f"{low:.1f}-{high:.1f}",
                "predicted": round(float(probabilities[mask].mean()), 3),
                "observed": round(float(labels[mask].mean()), 3),
                "count": int(mask.sum()),
            })
    return {
        "brierScore": round(float(brier_score_loss(labels, probabilities)), 4),
        "bins": bins,
    }


def fit_probability_calibrator(labels, probabilities):
    """Fit a sigmoid calibration map from strictly out-of-fold predictions."""
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    calibrator = LogisticRegression()
    calibrator.fit(np.asarray(probabilities).reshape(-1, 1), labels)
    return calibrator


def calibration_holdout_summary(oof_predictions: pd.DataFrame):
    """Evaluate calibration on the latest fold, fitting only on earlier folds."""
    latest_fold = oof_predictions["fold"].max()
    fit_rows = oof_predictions[oof_predictions["fold"] < latest_fold]
    test_rows = oof_predictions[oof_predictions["fold"] == latest_fold]
    calibrator = fit_probability_calibrator(
        fit_rows["label"], fit_rows["pred_prob"]
    )
    if calibrator is None or test_rows.empty:
        return None

    calibrated = calibrator.predict_proba(
        test_rows["pred_prob"].to_numpy().reshape(-1, 1)
    )[:, 1]
    return {
        "fitThroughFold": int(latest_fold - 1),
        "testFold": int(latest_fold),
        "rows": int(len(test_rows)),
        "rawBrierScore": round(
            float(brier_score_loss(test_rows["label"], test_rows["pred_prob"])), 4
        ),
        "calibratedBrierScore": round(
            float(brier_score_loss(test_rows["label"], calibrated)), 4
        ),
    }


def train_and_evaluate(df: pd.DataFrame):
    accuracies, aucs = [], []
    oof_predictions = []

    for fold_i, (train_df, test_df) in enumerate(
        walk_forward_splits(
            df, config.N_WALKFORWARD_SPLITS, reserve_final=False
        ), start=1
    ):
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
        accuracies.append(acc)
        aucs.append(auc)

        print(f"Fold {fold_i}: train={len(train_df)} rows, test={len(test_df)} rows "
              f"-> accuracy={acc:.3f}, AUC={auc:.3f}, "
              f"Brier={brier_score_loss(y_test, probs):.3f}")
        fold_predictions = test_df[["ticker", "forward_return"]].copy()
        fold_predictions["label"] = y_test
        fold_predictions["pred_prob"] = probs
        fold_predictions["fold"] = fold_i
        oof_predictions.append(fold_predictions)

    print(f"\nAverage walk-forward accuracy: {np.mean(accuracies):.3f}")
    print(f"Average walk-forward AUC:      {np.nanmean(aucs):.3f}")
    print(
        "\nReminder: >0.65 accuracy on your first run is a strong signal to "
        "double-check for leakage before celebrating."
    )
    if not oof_predictions:
        raise ValueError("Not enough dated feature rows for walk-forward validation.")
    oof = pd.concat(oof_predictions)
    print(f"Combined out-of-fold calibration: {calibration_summary(oof['label'], oof['pred_prob'])}")
    print(f"Calibration check on latest fold: {calibration_holdout_summary(oof)}")
    return oof


def train_final_model(df: pd.DataFrame, oof_predictions: pd.DataFrame):
    """Refit on all labeled history after evaluation for current live predictions."""
    X, y = df[FEATURE_COLUMNS], df["label"]
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        verbose=-1,
    )
    model.fit(X, y)
    calibrator = fit_probability_calibrator(
        oof_predictions["label"], oof_predictions["pred_prob"]
    )
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "probability_calibrator": calibrator,
        "calibration_validation": calibration_holdout_summary(oof_predictions),
    }, config.MODEL_PATH)
    print(f"\nFinal model trained on all {len(df)} labeled rows and saved to {config.MODEL_PATH}")


def main():
    df = load_features()
    development, _, test_start = split_final_test_period(df)
    print(
        f"Loaded {len(df)} rows across {df['ticker'].nunique()} tickers. "
        f"Development through {development.index.max().date()}; "
        f"reserved final period starts {test_start.date()}.\n"
    )

    print("=== Walk-forward validation (honest, out-of-sample estimate) ===")
    oof_predictions = train_and_evaluate(development)

    print("\n=== Training live model on all available labeled history ===")
    train_final_model(df, oof_predictions)

    print("\nNext step: python backtest/backtest.py")


if __name__ == "__main__":
    main()
