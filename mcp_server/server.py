"""
MCP server exposing the AI Stock Estimator as tools any MCP-compatible
client (e.g. Claude Desktop, Claude Code) can call directly -- ask
"predict NVDA" in a chat instead of running scripts by hand.

Setup:
  1. pip install -r requirements-mcp.txt
     (adds the mcp package on top of the main project's requirements.txt)
  2. Register this server with your MCP client -- see the README section
     "Using this model via MCP" for the exact config snippet.
  3. Ask things like "predict NVDA" or "what were the holdout backtest
     results?" and your assistant will call these tools directly.

IMPORTANT -- this server is READ-ONLY. None of these tools train,
retrain, or modify anything on disk. predict_stock always uses whatever
model is currently saved at models/model.pkl; run the training pipeline
yourself (see README) to update it. This server reflects whatever's on
disk at the moment it's asked -- it does not retrain in the background.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from mcp.server.mcpserver import MCPServer

import config
from serve.live_predict import predict_ticker

server = MCPServer("ai-stock-estimator")


# --- Core logic, kept separate from the @server.tool() decorators so it's
#     directly callable/testable without going through an MCP transport. ---

def _predict_stock(ticker: str) -> dict:
    if not config.MODEL_PATH.exists():
        return {"error": "No trained model found. Run models/train_model.py first."}
    bundle = joblib.load(config.MODEL_PATH)
    try:
        return predict_ticker(ticker.upper(), bundle)
    except Exception as e:
        return {"error": f"Could not predict {ticker.upper()}: {e}"}


def _list_tickers() -> dict:
    return {
        "training_tickers": config.TICKERS,
        "holdout_tickers": config.HOLDOUT_TICKERS,
    }


def _load_results(filename: str) -> dict:
    path = config.RESULTS_DIR / filename
    if not path.exists():
        return {
            "error": f"No results found at {path}. Run the matching "
                     f"backtest script first, then try again."
        }
    return json.loads(path.read_text())


# --- MCP tool registrations -------------------------------------------

@server.tool()
def predict_stock(ticker: str) -> dict:
    """
    Predict whether a stock's price will go up or down over the model's
    configured horizon, using live market data. Returns direction,
    confidence (%), latest close price, and the horizon in days. Works
    for any valid stock ticker, not just ones the model trained on --
    though predictions are most trustworthy for tickers similar to what
    it trained on. This is NOT financial advice -- see the project's
    validation results before trusting any prediction.
    """
    return _predict_stock(ticker)


@server.tool()
def list_tickers() -> dict:
    """List the stocks in the training universe and the separate holdout
    universe used to validate the model."""
    return _list_tickers()


@server.tool()
def get_training_backtest_results() -> dict:
    """
    Get the most recent backtest results on the training universe: win
    rate, compounded return, and how it compares to simply buying and
    holding the same stocks -- the real bar the model needs to clear.
    """
    return _load_results("training_backtest.json")


@server.tool()
def get_holdout_backtest_results() -> dict:
    """
    Get the most recent backtest results on the holdout universe --
    stocks the model never trained on. This is the most honest test of
    whether the model generalizes or is overfit to its training stocks.
    """
    return _load_results("holdout_backtest.json")


if __name__ == "__main__":
    server.run()
