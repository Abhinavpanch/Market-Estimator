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
import hmac
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from flask import Flask, g, jsonify, request, send_from_directory, session
from pymongo.errors import PyMongoError
from dotenv import dotenv_values, load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
secret_key = os.environ.get("FLASK_SECRET_KEY") or dotenv_values(
    PROJECT_ROOT / ".env"
).get("FLASK_SECRET_KEY")
import config
from dashboard.auth import get_user, register_auth_routes
from serve.live_predict import predict_ticker

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent))
app.config.update(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower()
    in {"1", "true", "yes"},
)
app.permanent_session_lifetime = timedelta(days=7)

_PUBLIC_AUTH_PATHS = {
    "/api/auth/signup",
    "/api/auth/verify",
    "/api/auth/login",
    "/api/auth/me",
}


@app.before_request
def require_verified_user():
    if (
        request.path.startswith("/api/auth/")
        and request.path != "/api/auth/me"
        and not app.secret_key
    ):
        return jsonify({"error": "Set FLASK_SECRET_KEY before using account features."}), 503
    if not request.path.startswith("/api/") or request.path in _PUBLIC_AUTH_PATHS:
        return
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Log in to access this resource."}), 401
    user = get_user(user_id)
    if not user:
        session.clear()
        return jsonify({"error": "Log in to access this resource."}), 401
    g.user = user
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            return jsonify({"error": "Invalid request token. Refresh and try again."}), 403


@app.errorhandler(RuntimeError)
def handle_configuration_error(error):
    app.logger.error("Dashboard configuration error: %s", error)
    return jsonify({"error": str(error)}), 503


@app.errorhandler(PyMongoError)
def handle_database_error(error):
    app.logger.exception("Dashboard database request failed")
    return jsonify({"error": "Account storage is unavailable. Check MongoDB configuration."}), 503


register_auth_routes(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/favicon.svg")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


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


@app.route("/api/results/final-period")
def final_period_results():
    return _load_json_or_404(config.RESULTS_DIR / "final_period.json")


@app.route("/api/tickers")
def tickers():
    return jsonify({
        "training": config.TICKERS,
        "holdout": config.HOLDOUT_TICKERS,
    })


@app.route("/api/predict/<ticker>")
def predict(ticker: str):
    ticker = ticker.upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", ticker):
        return jsonify({"error": "Enter a valid stock ticker symbol."}), 400
    if not config.MODEL_PATH.exists():
        return jsonify({"error": "No trained model found. Run models/train_model.py first."}), 404

    try:
        bundle = joblib.load(config.MODEL_PATH)
        result = predict_ticker(ticker, bundle)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Could not predict {ticker.upper()}: {e}"}), 400


if __name__ == "__main__":
    print("Starting dashboard server...")
    print(f"Account secret configured: {bool(app.secret_key)}")
    print("Open http://localhost:5000 in your browser.")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, port=5000)
