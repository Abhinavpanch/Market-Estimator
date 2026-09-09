"""
Local server powering the live results + prediction dashboard.

Serves three things:
  1. The dashboard.html page itself.
  2. Saved backtest results as JSON, read from results/*.json -- these
     files are written automatically every time you run backtest/backtest.py
     or backtest/test_holdout.py, so the dashboard always reflects your
     most recent run without any manual copy-pasting.
  3. Live predictions on demand via /api/predict/<ticker>, using the same
     trained model and feature logic as serve/live_predict.py. This
     fetches fresh price (and sentiment, if configured) data every time
     it's called, so it can take a few seconds per request.

Run with:
    python dashboard/server.py
Then open in a browser:
    http://localhost:5000
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from flask import Flask, jsonify, send_from_directory

import config
from serve.live_predict import predict_ticker

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


def _load_json_or_404(path: Path):
    if not path.exists():
        return jsonify({
            "error": f"No results yet at {path.name}. Run the matching backtest "
                     f"script first, then refresh this page."
        }), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/results/training")
def training_results():
    return _load_json_or_404(config.RESULTS_DIR / "training_backtest.json")


@app.route("/api/results/holdout")
def holdout_results():
    return _load_json_or_404(config.RESULTS_DIR / "holdout_backtest.json")


@app.route("/api/tickers")
def tickers():
    return jsonify({
        "training": config.TICKERS,
        "holdout": config.HOLDOUT_TICKERS,
    })


@app.route("/api/predict/<ticker>")
def predict(ticker: str):
    if not config.MODEL_PATH.exists():
        return jsonify({"error": "No trained model found. Run models/train_model.py first."}), 404

    bundle = joblib.load(config.MODEL_PATH)
    try:
        result = predict_ticker(ticker.upper(), bundle)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Could not predict {ticker.upper()}: {e}"}), 400


if __name__ == "__main__":
    print("Starting dashboard server...")
    print("Open http://localhost:5000 in your browser.")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=True, port=5000)
