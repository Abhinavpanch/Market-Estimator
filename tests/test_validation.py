import tempfile
import unittest
from pathlib import Path
import sys
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from backtest.backtest import (
    compute_equity_curve,
    compute_portfolio_equity_curve,
    enforce_non_overlapping,
    risk_metrics,
)
from dashboard.server import app
from features.build_features import add_label, merge_sentiment
from models.train_model import (
    calibration_holdout_summary,
    final_test_start,
    split_final_test_period,
    walk_forward_splits,
)
from serve.live_predict import (
    forecast_target_date,
    get_latest_sentiment,
    predict_ticker,
)


class ValidationTests(unittest.TestCase):
    def test_labels_match_next_open_entry_and_horizon_exit(self):
        frame = pd.DataFrame(
            {"open": np.arange(100.0, 111.0), "close": np.arange(100.0, 111.0)},
            index=pd.date_range("2024-01-01", periods=11),
        )

        labeled = add_label(frame, horizon=2)

        self.assertAlmostEqual(labeled["forward_return"].iloc[0], 103 / 101 - 1)
        self.assertEqual(labeled["label"].iloc[0], 1)
        self.assertTrue(labeled["forward_return"].tail(3).isna().all())

    def test_sentiment_is_lagged_one_market_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "TEST.csv"
            pd.DataFrame(
                {"news_sentiment": [0.8, -0.5], "news_volume": [2, 1]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            ).to_csv(path)
            frame = pd.DataFrame(
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
            )
            with patch.object(config, "NEWS_SENTIMENT_DIR", Path(temp_dir)):
                merged = merge_sentiment(frame, "TEST")

        self.assertEqual(merged["news_sentiment"].tolist(), [0.0, 0.8, -0.5])

    def test_walk_forward_purges_unavailable_outcomes(self):
        dates = pd.date_range("2020-01-01", periods=80)
        horizon_gap = config.PREDICTION_HORIZON_DAYS + 1
        frame = pd.DataFrame(
            {
                "ticker": "TEST",
                "label": 1,
                "forward_exit_date": pd.Series(dates, index=dates).shift(-horizon_gap),
            },
            index=dates,
        )

        for train, test in walk_forward_splits(frame, n_splits=3):
            test_start = test.index.min()
            self.assertLess(train.index.max(), test_start)
            self.assertLess(pd.to_datetime(train["forward_exit_date"]).max(), test_start)
            self.assertLess(
                pd.to_datetime(test["forward_exit_date"]).max(),
                final_test_start(frame),
            )

    def test_reserved_final_period_is_excluded_and_training_is_purged(self):
        dates = pd.date_range("2020-01-01", periods=100)
        frame = pd.DataFrame(
            {
                "forward_exit_date": pd.Series(dates, index=dates).shift(-3),
                "label": 1,
            },
            index=dates,
        ).dropna()

        development, test, test_start = split_final_test_period(frame)

        self.assertLess(development.index.max(), test_start)
        self.assertLess(pd.to_datetime(development["forward_exit_date"]).max(), test_start)
        self.assertGreaterEqual(test.index.min(), test_start)
        self.assertLessEqual(pd.to_datetime(test["forward_exit_date"]).max(), frame.index.max())

    def test_daily_equity_curve_marks_open_position_to_market(self):
        dates = pd.date_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {
                "ticker": "TEST",
                "open": [100.0, 110.0, 90.0, 120.0, 100.0],
            },
            index=dates,
        )
        signals = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "forward_exit_date": [dates[3]],
            },
            index=[dates[0]],
        )
        trades = signals.copy()
        trades["net_return"] = [0.09]

        equity = compute_portfolio_equity_curve(
            prices, signals, ["TEST"], dates[0], dates[3], cost_bps=0
        )
        metrics = risk_metrics(equity, trades)

        self.assertLess(equity.loc[dates[2]], equity.loc[dates[0]])
        self.assertGreater(equity.loc[dates[3]], equity.loc[dates[0]])
        self.assertLess(metrics["maxDrawdown"], 0)

    def test_live_sentiment_uses_prior_session_and_runtime_key(self):
        sentiment_date = pd.Timestamp("2025-03-07")
        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}), \
             patch(
                 "serve.live_predict.fetch_company_news", return_value=[{"test": True}]
             ) as fetch, \
             patch(
                 "serve.live_predict.aggregate_daily_sentiment",
                 return_value=pd.DataFrame(
                     {"news_sentiment": [0.4], "news_volume": [2]},
                     index=[sentiment_date],
                 ),
             ):
            result = get_latest_sentiment("TEST", sentiment_date)

        fetch.assert_called_once_with("TEST", "2025-03-07", "2025-03-07", "test-key")
        self.assertEqual(result, (0.4, 2.0))

    def test_inactive_ticker_allocation_remains_cash(self):
        trades = pd.DataFrame(
            {"ticker": ["A"], "net_return": [0.1]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        self.assertAlmostEqual(compute_equity_curve(trades, n_tickers=2), 1.05)

    def test_trades_use_exact_forward_exit_date_for_overlap(self):
        signals = pd.DataFrame(
            {
                "ticker": ["TEST", "TEST", "TEST"],
                "forward_exit_date": pd.to_datetime(
                    ["2024-01-16", "2024-01-17", "2024-01-18"]
                ),
            },
            index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-16"]),
        )

        trades = enforce_non_overlapping(signals)

        self.assertEqual(list(trades.index), list(signals.index[[0, 2]]))

    def test_calibration_holdout_uses_prior_folds(self):
        predictions = pd.DataFrame(
            {
                "fold": [1, 1, 2, 2],
                "label": [0, 1, 0, 1],
                "pred_prob": [0.2, 0.8, 0.4, 0.6],
            }
        )

        summary = calibration_holdout_summary(predictions)

        self.assertEqual(summary["fitThroughFold"], 1)
        self.assertEqual(summary["testFold"], 2)
        self.assertEqual(summary["rows"], 2)

    def test_prediction_includes_recent_candles_and_visual_direction(self):
        class Model:
            feature_importances_ = np.array([5])

            def predict_proba(self, features):
                return np.array([[0.2, 0.8]])

        history = [
            {"date": "2024-01-02", "open": 100, "high": 103, "low": 99, "close": 102},
            {"date": "2024-01-03", "open": 102, "high": 104, "low": 101, "close": 103},
        ]
        latest = pd.DataFrame({
            "volatility_10": [0.02],
            "news_sentiment": [0.317],
            "news_volume": [2],
        })
        bundle = {"model": Model(), "feature_columns": ["volatility_10"]}

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}), \
             patch(
                 "serve.live_predict.get_latest_features",
                 return_value=(latest, 103.0, history),
             ):
            result = predict_ticker("TEST", bundle)

        self.assertEqual(result["history"], history)
        self.assertEqual(result["direction"], "UP")
        self.assertGreater(result["illustrative_close"], result["latest_close"])
        self.assertTrue(result["sentiment_key_configured"])
        self.assertTrue(result["sentiment_articles"])
        self.assertEqual(result["sentiment_score"], 0.317)
        self.assertEqual(result["sentiment_article_count"], 2)
        self.assertFalse(result["sentiment_model_uses_feature"])
        self.assertEqual(result["horizon_days"], 5)
        self.assertEqual(result["horizon_calendar_days"], 7)
        self.assertEqual(
            forecast_target_date(date(2026, 11, 2)),
            "2026-11-09",
        )

    def test_api_accepts_unlisted_ticker_and_rejects_invalid_symbols(self):
        with patch.dict(app.config, {"SECRET_KEY": "test-secret"}), \
             patch("dashboard.server.get_user", return_value={"_id": "test", "verified": True}):
            client = app.test_client()
            with client.session_transaction() as browser_session:
                browser_session["user_id"] = "test"
                browser_session["csrf_token"] = "test-csrf"
            with patch("dashboard.server.config.MODEL_PATH") as model_path, \
                 patch("dashboard.server.joblib.load", return_value={}), \
                 patch("dashboard.server.predict_ticker", return_value={"ticker": "TEST"}) as predict:
                model_path.exists.return_value = True
                response = client.get("/api/predict/TEST")
                invalid_response = client.get("/api/predict/%3Cscript%3E")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ticker": "TEST"})
        self.assertEqual(invalid_response.status_code, 400)
        predict.assert_called_once()

    def test_api_serves_reserved_period_results(self):
        with patch.dict(app.config, {"SECRET_KEY": "test-secret"}), \
             patch("dashboard.server.get_user", return_value={"_id": "test", "verified": True}):
            client = app.test_client()
            with client.session_transaction() as browser_session:
                browser_session["user_id"] = "test"
                browser_session["csrf_token"] = "test-csrf"
            with tempfile.TemporaryDirectory() as temp_dir:
                results_path = Path(temp_dir)
                (results_path / "final_period.json").write_text(
                    '{"summary":{"testStart":"2025-01-01"}}'
                )
                with patch.object(config, "RESULTS_DIR", results_path):
                    response = client.get("/api/results/final-period")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["summary"]["testStart"], "2025-01-01")


if __name__ == "__main__":
    unittest.main()
