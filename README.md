# AI Stock Estimator — Starter Project

A minimal, working scaffold for building and training your own stock-direction
prediction model, matching the pipeline we outlined:

fetch data → build features → train model → backtest → serve live predictions

This is a **starting point**, not a finished product. It uses a simple,
honest baseline (gradient-boosted trees on technical-indicator features) so
you have something real to run, test, and improve — rather than a black box.

**This is not financial advice, and nothing here is a guarantee of predictive
accuracy.** Treat any promising-looking result with suspicion until you've
checked for data leakage (see `models/train_model.py`).

---

## Project structure

```
ai_stock_estimator/
├── requirements.txt          # Python dependencies
├── config.py                 # Central settings (tickers, horizon, paths)
├── data/
│   └── fetch_data.py         # Downloads historical OHLCV data
├── features/
│   └── build_features.py     # Turns raw prices into model features
├── models/
│   └── train_model.py        # Walk-forward training + evaluation
├── backtest/
│   └── backtest.py           # Simulates trading on model predictions
└── serve/
    └── live_predict.py       # Loads the trained model, predicts on live data
```

## Setup

```bash
cd ai_stock_estimator
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline, in order

```bash
python data/fetch_data.py             # 1. downloads historical price data
python data/fetch_news_sentiment.py   # 2. downloads + scores historical news (optional, see below)
python features/build_features.py    # 3. builds features to data/features.csv
python models/train_model.py         # 4. trains model, saves to models/model.pkl
python backtest/backtest.py          # 5. simulates trading, prints performance
python serve/live_predict.py         # 6. one-shot live prediction using latest data
```

## What changed since the first version

**Wider universe.** `config.TICKERS` now spans 32 stocks across tech,
finance, healthcare, energy, consumer staples/discretionary, industrials,
utilities, and materials -- not just 5 mega-cap tech names. This matters
because the first version's backtest looked great almost entirely because
of one stock's (NVDA) historic run; a wider, more diverse universe makes
it much harder for one stock's story to carry the whole result.
`config.HOLDOUT_TICKERS` is a separate 10-stock set across similarly
varied sectors, with zero overlap with the training list.

**News sentiment feature.** `news_sentiment` and `news_volume` are now
part of the feature set (`models/train_model.py FEATURE_COLUMNS`). Two
honest limitations worth knowing:

- **Free historical news coverage is best-effort.** Most free news APIs
  only cover the last month; this project uses Finnhub's free tier
  because it supports historical date ranges, but coverage for older
  dates or less-followed tickers may still be thin. Run
  `data/fetch_news_sentiment.py` and check the printed article counts --
  a ticker with very few articles won't get much real signal from this
  feature, and it'll mostly just be 0.
- **VADER (the sentiment scorer used here) is general-purpose, not
  finance-tuned.** It catches clearly emotional language well but tends
  to score neutral-sounding financial phrasing (like "beats estimates")
  close to 0 even when it's actually good news. A finance-tuned model
  (e.g. FinBERT) would likely be a meaningfully better signal here if
  you want to take this further.

**Getting a free Finnhub API key** (needed for sentiment; everything
else works without it):
1. Sign up at https://finnhub.io/register
2. Set it as an environment variable before running the sentiment scripts:
   - Windows (PowerShell): `$env:FINNHUB_API_KEY="your_key_here"`
   - Mac/Linux (bash): `export FINNHUB_API_KEY="your_key_here"`

If you skip this, the pipeline still runs -- `news_sentiment` and
`news_volume` will just default to 0 everywhere, same as before this
feature existed.

## Live dashboard (auto-updating results + live predictions)

```bash
python dashboard/server.py
```

Then open **http://localhost:5000** in your browser. This gives you:

- **Auto-updating backtest results.** Every time you run
  `backtest/backtest.py` or `backtest/test_holdout.py`, they now save a
  JSON snapshot to `results/`. The dashboard reads those files live --
  just refresh the browser page after a new run to see updated numbers.
  No copy-pasting numbers into HTML by hand anymore.
- **Live, on-demand predictions.** Pick any stock from the dropdown and
  click Predict -- it fetches fresh price data and (if you set up a
  Finnhub key) recent news sentiment, runs it through your trained model,
  and shows direction, confidence, and latest price. Check "Auto-refresh
  every 30s" to keep it updating continuously.

**Requirements before this works:**
- `models/model.pkl` must exist (run `models/train_model.py` first) --
  needed for live predictions.
- `results/training_backtest.json` / `results/holdout_backtest.json` must
  exist (run `backtest/backtest.py` / `backtest/test_holdout.py`) --
  needed for the results sections. If they don't exist yet, the dashboard
  shows a clear message telling you which script to run.

Leave the server running in its own terminal window while you use the
dashboard; open a second terminal for running the other scripts.

## Retraining efficiently (one command)

Instead of running fetch/build/train/backtest scripts one at a time,
`retrain_all.py` runs the whole pipeline in the correct order:

```bash
python retrain_all.py                        # price data -> train -> backtest
python retrain_all.py --sentiment            # also refreshes news sentiment
python retrain_all.py --holdout              # also re-validates on holdout tickers
python retrain_all.py --sentiment --holdout  # everything
```

Run this any time you've changed `config.TICKERS`, want fresher data, or
just want to retrain on a schedule (e.g. weekly) without remembering the
exact step order. If `dashboard/server.py` is running, restart it
afterward so it picks up the new model and results.

## Using this model via MCP

MCP (Model Context Protocol) lets an AI assistant call this project's
model directly as a tool -- e.g. asking "predict NVDA" instead of running
scripts by hand. `mcp_server/server.py` exposes four read-only tools:

- `predict_stock(ticker)` -- live prediction for any ticker
- `list_tickers()` -- the training and holdout universes
- `get_training_backtest_results()` -- latest training backtest summary
- `get_holdout_backtest_results()` -- latest holdout test summary

This server never trains or modifies anything -- it only reads
`models/model.pkl` and the `results/*.json` files that already exist on
disk. Run `retrain_all.py` (or the individual scripts) separately to
update what it serves.

**Setup:**

```bash
pip install -r requirements-mcp.txt
```

**Registering with Claude Desktop or Claude Code:** add this to your
MCP client's config file (for Claude Desktop, that's
`claude_desktop_config.json` -- see
https://modelcontextprotocol.io/quickstart/user for its exact location
on your OS):

```json
{
  "mcpServers": {
    "ai-stock-estimator": {
      "command": "py",
      "args": ["C:\\path\\to\\ai_stock_estimator\\mcp_server\\server.py"]
    }
  }
}
```

Replace the path with wherever you unzipped this project (use the full
absolute path, and keep the double backslashes on Windows). Restart your
MCP client after editing the config. You should then be able to ask it
things like "what's the AI stock estimator's prediction for AAPL?" and
it will call `predict_stock` directly.

## Validating a backtest result before trusting it

A backtest that beats buy-and-hold is a promising sign, not proof. Before
trusting any result, run these checks:

```bash
python data/fetch_holdout_data.py           # fetch stocks the model NEVER trained on
python features/build_holdout_features.py   # build features for them
python backtest/test_holdout.py             # does the edge survive on unseen stocks?
python backtest/threshold_sweep.py          # is the result stable across thresholds?
```

`backtest/backtest.py` itself now also prints:
- a **per-ticker breakdown** (is the return broadly spread, or all from one stock?)
- a **random-entry baseline** (does randomly-timed trading do almost as well as the model?)

If the holdout test shows a similar edge, the threshold sweep is stable
across nearby values, and the model clearly beats the random baseline --
that's a much stronger signal than a single backtest number. If any of
those checks fail, treat the original result as likely regime-specific or
overfit rather than a real, durable edge.

### Testing on a different historical period

Set `START_DATE` and `END_DATE` in `config.py` (format `"YYYY-MM-DD"`) to
pull a specific historical window instead of the last 5 years, e.g. to see
if results hold up in 2015-2020 rather than only the recent tech bull run.
Re-run the full pipeline (steps 1-4 above) after changing these -- results
will overwrite the previous run's files, so note down numbers you want to
compare before switching windows.


## What to change first

- **`config.py`** — set your own ticker list and prediction horizon.
- **`features/build_features.py`** — this is where your real "edge" will come
  from. Start by adding/removing indicators and see what helps.
- **`models/train_model.py`** — swap the model, tune hyperparameters, but do
  NOT touch the walk-forward split logic without understanding why it's there
  (it exists specifically to prevent the model from "seeing the future").

## Honest expectations

A model that shows >70% directional accuracy on its first run is far more
likely to have a data leak than a real edge. Real, defensible edges in this
space are usually small (55-58% directional accuracy is already notable) —
the value comes from consistency and good risk management, not from any
single "lucky" model.
