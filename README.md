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

## Tests

```bash
python -m unittest discover -s tests -v
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
- Live predictions read `FINNHUB_API_KEY` when each prediction is
  requested and fetch sentiment for the prior market session, matching
  training's one-session lag. Set the variable in the same terminal
  session that starts the dashboard or runs predictions.

**Getting a free Finnhub API key** (needed for sentiment; everything
else works without it):
1. Sign up at https://finnhub.io/register
2. Set it as an environment variable before running the sentiment scripts:
   - Windows (PowerShell): `$env:FINNHUB_API_KEY="your_key_here"`
   - Mac/Linux (bash): `export FINNHUB_API_KEY="your_key_here"`

Or save it once in a `.env` file in the project root:

```text
FINNHUB_API_KEY=your_actual_key
```

`.env` is ignored by Git and loaded automatically by the project. Do not
commit it or share its contents. Restart any running dashboard after
changing the file.

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
  The prediction card also reports whether Finnhub returned articles or
  the key is missing for the current process.

**Requirements before this works:**
- `models/model.pkl` must exist (run `models/train_model.py` first) --
  needed for live predictions.
- `results/training_backtest.json` / `results/holdout_backtest.json` must
  exist (run `backtest/backtest.py` / `backtest/test_holdout.py`) --
  needed for the results sections. If they don't exist yet, the dashboard
  shows a clear message telling you which script to run.

### Accounts and personal watchlists

The dashboard requires an email-verified account. Sign up with a username,
email, and password, enter the one-time code sent to your email, then log in
with either your username or email and password. Each account has its own
watchlist; adding or removing a ticker only affects that signed-in account.

Configure MongoDB Atlas and an SMTP email provider in the project-root
`.env` file (which is ignored by Git):

```text
MONGO_URI=mongodb+srv://<database-user>:<password>@<cluster>/<database>
MONGO_DB_NAME=ai_stock_estimator
FLASK_SECRET_KEY=<long-random-secret>
SMTP_HOST=<your-smtp-host>
SMTP_PORT=587
SMTP_USERNAME=<your-smtp-username>
SMTP_PASSWORD=<your-smtp-password>
MAIL_FROM=<verified-sender-email>
SESSION_COOKIE_SECURE=false
```

Use your Atlas database user's credentials in `MONGO_URI` (not an Atlas
account/API key), and allow the app host in Atlas Network Access. Keep these
values private and never paste them into source control or chat. For an
HTTPS deployment, set `SESSION_COOKIE_SECURE=true`; leave it `false` only
for local HTTP development. Verification codes expire after 10 minutes.

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
python backtest/model_comparison.py         # compare LightGBM, logistic, technical-only
python backtest/final_period.py             # evaluate the reserved latest 15% period
```

`backtest/backtest.py` itself now also prints:
- a **per-ticker breakdown** (is the return broadly spread, or all from one stock?)
- a **random-entry baseline** (does randomly-timed trading do almost as well as the model?)
- mark-to-market **maximum drawdown, annualized volatility, and Sharpe**

The pipeline uses expanding walk-forward periods with a purge gap so that
training labels cannot reach into the test period. The latest 15% of dates
are excluded from development folds, model comparisons, and the final live
model evaluation. After evaluation, the live model is refit on all labeled
history, including the reserved dates; consequently, that historical period
is not an independent test of the deployed fit. `model_comparison.py`
compares a small, predeclared set of baselines using development folds only.
Holdout testing also
trains only on earlier dates from the training tickers before testing the
unseen stocks in each later period. Metrics include Brier score and
calibration bins; compare predicted and observed up rates rather than
treating model confidence as certainty.

Run `backtest/final_period.py` once after development choices are fixed. It
reports the reserved segment against buy-and-hold and random timing, but
this historical period appeared in earlier exploratory runs, so it is
**not** an untouched independent test. A genuinely independent evaluation
requires recording predictions prospectively and not tuning or retraining
against that future period.

Signals are computed after the close, entered at the next session's open,
and measured open-to-open over the configured horizon. News sentiment is
lagged by one trading session. Backtests subtract a configurable 25 bps
round-trip estimate for fees and slippage and show a 10/25/50 bps
sensitivity. Replace this estimate with your broker, exchange, taxes, and
market-specific costs before drawing conclusions. Partial fills, market
impact, and taxes are not modeled.
Risk metrics mark positions to market at daily opens; they do not capture
intraday drawdown and their annualization is approximate across tickers
with different trading calendars. They are diagnostics, not a portfolio
risk model.

These are still historical simulations, not proof of a durable edge. A
positive holdout result can be noise, and market conditions may change.
This project uses a current, manually maintained ticker list rather than
point-in-time historical constituents, so survivorship bias remains. In
the current expanded-history run, nonzero news sentiment covers about 0.3%
of all feature rows and about 2% of rows in the most recent five years.
The model may assign little influence to sentiment; live predictions show
the fetched score, article count, and whether the fitted model learned to
use those features. Cached market data is not refreshed automatically;
rerun the fetch scripts before a new evaluation.

### Rebuild and validate using already downloaded data

The revised labels and validation require rebuilding the feature files,
then retraining and rerunning both backtests. These commands use existing
`data/raw/` CSVs and do not fetch market data:

```bash
python features/build_features.py
python features/build_holdout_features.py
python models/train_model.py
python backtest/backtest.py
python backtest/test_holdout.py
python backtest/model_comparison.py
python backtest/final_period.py
```

Walk-forward folds report results across multiple out-of-time periods.
Avoid selecting a threshold or changing the feature set based on the same
periods you later report as independent validation.


## What to change first

- **`config.py`** — set your own ticker list and prediction horizon.
- The default prediction target is five market sessions (approximately
  seven calendar days) ahead, using the latest available price data. The
  chart's illustrative target date is seven calendar days from today; market
  holidays and delayed quotes can make that differ from the actual fifth
  trading session. The projected line is not a model price target.
- Historical price fetching uses `HISTORY_PERIOD = "max"` to request all
  history available from the data provider. Actual history varies by
  ticker. More history does not remove survivorship bias or guarantee
  better predictions; after changing data range or horizon, rebuild
  features and retrain before using the model.
- **`features/build_features.py`** — this is where your real "edge" will come
  from. Start by adding/removing indicators and see what helps.
- **`models/train_model.py`** — swap the model, tune hyperparameters, but do
  NOT touch the walk-forward split logic without understanding why it's there
  (it exists specifically to prevent the model from "seeing the future").

## Honest expectations

A high accuracy or attractive historical return does not imply a
deployable edge. Check calibration, out-of-time and unseen-stock periods,
the random-entry baseline, and the cost sensitivity together. None
guarantees future results.
