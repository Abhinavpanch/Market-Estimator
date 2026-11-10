"""
Central configuration for the AI Stock Estimator starter project.
Edit this file first -- almost everything else reads from here.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")


def _load_ticker_list(filename: str) -> list:
    """
    Reads a plain-text ticker file: one symbol per line, blank lines and
    lines starting with # ignored. This means adding/removing a stock is
    just editing a text file (tickers.txt / holdout_tickers.txt) -- no
    Python syntax, quotes, or commas needed.
    """
    path = ROOT_DIR / filename
    tickers = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line)
    return tickers


# --- Universe -----------------------------------------------------------
# Loaded from tickers.txt / holdout_tickers.txt at the project root --
# edit those files to change the universe, not this one.
TICKERS = _load_ticker_list("tickers.txt")

# Stocks the model NEVER trains or tests on during normal development.
# Used only by backtest/test_holdout.py to check whether the model
# generalizes beyond the training universe, or just memorized it.
HOLDOUT_TICKERS = _load_ticker_list("holdout_tickers.txt")

_overlap = set(TICKERS) & set(HOLDOUT_TICKERS)
if _overlap:
    print(
        f"WARNING: {sorted(_overlap)} appear in BOTH tickers.txt and "
        f"holdout_tickers.txt. The holdout test is only meaningful if "
        f"these lists never overlap -- remove these from one file."
    )

# --- Prediction target ----------------------------------------------------
# Approximate a 7-calendar-day target with 5 trading sessions.
PREDICTION_HORIZON_DAYS = 5
PREDICTION_CALENDAR_DAYS = 7

# Round-trip fees and slippage are ticker/broker dependent. This is a
# conservative, adjustable estimate, not a substitute for your actual costs.
ROUND_TRIP_COST_BPS = 25

# How much history to pull for training (longer = more data, slower to fetch)
HISTORY_PERIOD = "max"      # fetch the full history available from the data provider
HISTORY_INTERVAL = "1d"     # daily bars -- keep this for the starter project

# Optional: set both to fetch a SPECIFIC historical window instead of the
# last HISTORY_PERIOD (e.g. to test on 2015-2020 instead of the last 5
# years). Leave both as None to use HISTORY_PERIOD as normal. Format:
# "YYYY-MM-DD".
START_DATE = None
END_DATE = None

# --- Paths ----------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_PATH = DATA_DIR / "features.csv"
HOLDOUT_FEATURES_PATH = DATA_DIR / "holdout_features.csv"
NEWS_SENTIMENT_DIR = DATA_DIR / "news_sentiment"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
RESULTS_DIR = ROOT_DIR / "results"  # JSON snapshots the dashboard reads from

# --- Walk-forward validation ----------------------------------------------
# Number of folds for walk-forward validation (see models/train_model.py)
N_WALKFORWARD_SPLITS = 5

# Preserve the latest portion of the timeline for a final time-based test.
FINAL_TEST_FRACTION = 0.15

for d in (RAW_DATA_DIR, MODELS_DIR, NEWS_SENTIMENT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
