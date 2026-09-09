"""
Simulates a simple trading strategy built on the model's predictions, using
the same walk-forward folds as training so results stay honest (no lookahead).

Strategy (deliberately simple -- a starting point, not a recommendation):
  - If the model predicts "up" with probability > CONFIDENCE_THRESHOLD,
    buy at today's close and sell PREDICTION_HORIZON_DAYS later.
  - Otherwise, stay in cash.
  - Each trade pays TRANSACTION_COST_BPS (round-trip) to approximate real
    trading costs -- a model can look accurate and still lose money once
    this is included.

This does NOT account for slippage, taxes, position sizing across multiple
simultaneous trades, or market impact. Treat it as a sanity check, not a
real P&L estimate.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import walk_forward_splits, load_features

CONFIDENCE_THRESHOLD = 0.55   # only trade when the model is reasonably confident
TRANSACTION_COST_BPS = 10     # 10 basis points = 0.10% round-trip cost, adjust to your broker


def simulate_fold(train_df, test_df, threshold: float = CONFIDENCE_THRESHOLD):
    from lightgbm import LGBMClassifier
    from models.train_model import FEATURE_COLUMNS

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, verbose=-1)
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    # Approximate forward return over the prediction horizon per ticker
    test_df = test_df.sort_values(["ticker", test_df.index.name or "date"])
    test_df["future_return"] = test_df.groupby("ticker")["close"].pct_change(
        config.PREDICTION_HORIZON_DAYS
    ).shift(-config.PREDICTION_HORIZON_DAYS)

    signals = test_df[test_df["pred_prob"] > threshold].dropna(subset=["future_return"])
    if signals.empty:
        return None

    cost = TRANSACTION_COST_BPS / 10_000
    signals = signals.copy()
    signals["net_return"] = signals["future_return"] - cost

    return signals[["ticker", "pred_prob", "future_return", "net_return"]]


def enforce_non_overlapping(signals: pd.DataFrame) -> pd.DataFrame:
    """
    A model can fire a new signal on a ticker before the previous
    PREDICTION_HORIZON_DAYS trade on that same ticker has closed. With one
    real account, you can't hold unlimited overlapping positions in the
    same stock -- so this keeps only the FIRST signal per ticker, then
    skips any further signals on that ticker until the horizon has passed.
    This is what makes the resulting trade list something you could
    actually have taken with one pot of capital.
    """
    signals = signals.sort_index()
    kept_rows = []

    for ticker, group in signals.groupby("ticker"):
        next_allowed_date = None
        for date, row in group.iterrows():
            if next_allowed_date is not None and date < next_allowed_date:
                continue  # still "in a trade" on this ticker, skip
            kept_rows.append(row)
            next_allowed_date = date + pd.Timedelta(days=config.PREDICTION_HORIZON_DAYS)

    return pd.DataFrame(kept_rows)


def compute_equity_curve(trades: pd.DataFrame, n_tickers: int) -> float:
    """
    Compounds returns assuming capital is split evenly across tickers
    up front (e.g. 5 tickers -> 20% of capital each), and each ticker's
    slice compounds independently through its own non-overlapping trades.
    Returns the final portfolio multiple (e.g. 1.35 = +35% total).
    """
    starting_share = 1.0 / n_tickers
    final_value = 0.0

    for ticker, group in trades.groupby("ticker"):
        equity = starting_share
        for net_return in group.sort_index()["net_return"]:
            equity *= (1 + net_return)
        final_value += equity

    return final_value


def compute_per_ticker_stats(trades: pd.DataFrame, n_tickers: int) -> list:
    """
    Returns a list of {ticker, trades, win_rate, return} dicts -- the raw
    data behind the per-ticker breakdown. Used both for printing to the
    console and for saving to JSON for the dashboard.
    """
    starting_share = 1.0 / n_tickers
    stats = []

    for ticker, group in trades.groupby("ticker"):
        group = group.sort_index()
        win_rate = (group["net_return"] > 0).mean()
        equity = starting_share
        for r in group["net_return"]:
            equity *= (1 + r)
        ticker_return = (equity / starting_share) - 1
        stats.append({
            "ticker": ticker,
            "trades": int(len(group)),
            "winRate": round(float(win_rate) * 100, 1),
            "return": round(float(ticker_return) * 100, 1),
        })

    return stats


def print_per_ticker_breakdown(trades: pd.DataFrame, n_tickers: int) -> None:
    """
    Shows win rate, trade count, and compounded return SEPARATELY for each
    ticker. If almost all of the portfolio's outperformance comes from one
    stock, that's a concentration risk, not broad model skill.
    """
    print("\nPer-ticker breakdown:")
    print(f"{'TICKER':8s}{'TRADES':>8s}{'WIN RATE':>10s}{'RETURN':>12s}")
    for row in compute_per_ticker_stats(trades, n_tickers):
        print(f"{row['ticker']:8s}{row['trades']:>8d}{row['winRate']:>8.1f}% {row['return']:>10.1f}%")


def simulate_random_baseline(df: pd.DataFrame, trades: pd.DataFrame, n_tickers: int,
                              n_runs: int = 20, seed: int = 42) -> float:
    """
    Takes the SAME NUMBER of trades per ticker that the model actually
    took, but at randomly chosen (non-overlapping) entry dates instead of
    model-chosen ones. Averaged over several random runs. If this random
    strategy performs similarly to the model, the model isn't adding real
    timing skill -- it's just benefiting from being long these stocks
    during a generally rising period, same as random entries would.
    """
    rng = np.random.default_rng(seed)
    cost = TRANSACTION_COST_BPS / 10_000
    horizon = config.PREDICTION_HORIZON_DAYS
    trades_per_ticker = trades.groupby("ticker").size().to_dict()

    run_results = []
    for _ in range(n_runs):
        starting_share = 1.0 / n_tickers
        final_value = 0.0

        for ticker, n_trades_needed in trades_per_ticker.items():
            ticker_df = df[df["ticker"] == ticker].sort_index()
            ticker_df = ticker_df.dropna(subset=["close"])
            available_dates = ticker_df.index.to_list()

            equity = starting_share
            taken = 0
            used_ranges = []  # (start_idx, end_idx) already claimed, avoid overlap
            shuffled_starts = rng.permutation(len(available_dates))
            for start_idx in shuffled_starts:
                if taken >= n_trades_needed:
                    break
                exit_pos = start_idx + horizon
                if exit_pos >= len(available_dates):
                    continue
                if any(start_idx < end and exit_pos > start for start, end in used_ranges):
                    continue  # overlaps a trade already taken on this ticker
                entry_date = available_dates[start_idx]
                entry_price = ticker_df.loc[entry_date, "close"]
                exit_price = ticker_df["close"].iloc[exit_pos]
                trade_return = (exit_price / entry_price) - 1 - cost
                equity *= (1 + trade_return)
                used_ranges.append((start_idx, exit_pos))
                taken += 1

            final_value += equity

        run_results.append(final_value)

    return float(np.mean(run_results))


def compute_buy_and_hold_benchmark(df: pd.DataFrame, tickers: list, test_start_dates: list) -> float:
    """
    What you'd have earned just buying and holding these same tickers,
    equally weighted, over the same overall test window -- no model, no
    trading. This is the bar the model actually needs to clear to be
    worth using instead of doing nothing.
    """
    earliest_test_date = min(test_start_dates)
    latest_date = df.index.max()

    starting_share = 1.0 / len(tickers)
    final_value = 0.0

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker]
        ticker_df = ticker_df[ticker_df.index >= earliest_test_date]
        if len(ticker_df) < 2:
            continue
        start_price = ticker_df["close"].iloc[0]
        end_price = ticker_df["close"].iloc[-1]
        final_value += starting_share * (end_price / start_price)

    return final_value


def main():
    df = load_features()
    all_signals = []
    test_start_dates = []

    for fold_i, (train_df, test_df) in enumerate(
        walk_forward_splits(df, config.N_WALKFORWARD_SPLITS), start=1
    ):
        test_start_dates.append(test_df.index.min())
        result = simulate_fold(train_df, test_df)
        if result is not None:
            result["fold"] = fold_i
            all_signals.append(result)

    if not all_signals:
        print("No trades were taken -- try lowering CONFIDENCE_THRESHOLD.")
        return

    raw_signals = pd.concat(all_signals)
    trades = enforce_non_overlapping(raw_signals)

    win_rate = (trades["net_return"] > 0).mean()
    avg_return = trades["net_return"].mean()
    total_trades = len(trades)
    n_tickers = trades["ticker"].nunique()
    portfolio_multiple = compute_equity_curve(trades, n_tickers)

    benchmark_multiple = compute_buy_and_hold_benchmark(
        df, config.TICKERS, test_start_dates
    )

    print(f"Raw signals fired:            {len(raw_signals)}")
    print(f"Non-overlapping trades taken:  {total_trades}  "
          f"(one position per ticker at a time)")
    print(f"Win rate (after costs):        {win_rate:.1%}")
    print(f"Avg return per trade:          {avg_return:.2%}")
    print(f"Compounded portfolio return:    {(portfolio_multiple - 1):.2%}  "
          f"(capital split evenly across {n_tickers} tickers)")
    print(f"\nBuy-and-hold benchmark:        {(benchmark_multiple - 1):.2%}  "
          f"(just holding the same {len(config.TICKERS)} tickers, no model, no trading)")

    edge = (portfolio_multiple - 1) - (benchmark_multiple - 1)
    if edge > 0:
        print(f"Model beat buy-and-hold by:    {edge:.2%}")
    else:
        print(f"Model UNDERPERFORMED buy-and-hold by: {abs(edge):.2%}")
        print(
            "This means the model's apparent return is likely just riding "
            "the same underlying trend as buy-and-hold, not adding real skill."
        )

    print_per_ticker_breakdown(trades, n_tickers)

    print("\nRunning random-entry baseline (same trade counts, random timing, 20 runs)...")
    random_multiple = simulate_random_baseline(df, trades, n_tickers)
    random_edge = (random_multiple - 1) - (benchmark_multiple - 1)
    print(f"Random-entry avg return:        {(random_multiple - 1):.2%}  "
          f"(vs. buy-and-hold {(benchmark_multiple - 1):.2%})")

    if edge > 0 and (portfolio_multiple - 1) > (random_multiple - 1) * 1.2:
        print(
            f"\nModel ({(portfolio_multiple - 1):.1%}) clearly beats random timing "
            f"({(random_multiple - 1):.1%}) -- more likely a real, if modest, edge."
        )
    else:
        print(
            f"\nModel ({(portfolio_multiple - 1):.1%}) is close to or worse than random "
            f"timing ({(random_multiple - 1):.1%}) -- the model's outperformance vs. "
            "buy-and-hold may just be luck / regime, not real skill."
        )

    print(
        "\nStill simplified -- no slippage, no partial fills, and equal "
        "capital split across tickers rather than real position sizing."
    )

    save_results_json(trades, n_tickers, total_trades, win_rate, avg_return,
                       portfolio_multiple, benchmark_multiple, random_multiple)


def save_results_json(trades, n_tickers, total_trades, win_rate, avg_return,
                       portfolio_multiple, benchmark_multiple, random_multiple):
    """
    Writes results/training_backtest.json -- a snapshot the dashboard
    server reads and serves. Overwritten every time this script runs, so
    the dashboard always reflects your most recent backtest.
    """
    import json
    from datetime import datetime

    output = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "trades": total_trades,
            "winRate": round(float(win_rate) * 100, 1),
            "avgReturnPerTrade": round(float(avg_return) * 100, 2),
            "modelReturn": round((portfolio_multiple - 1) * 100, 2),
            "benchmarkReturn": round((benchmark_multiple - 1) * 100, 2),
            "randomReturn": round((random_multiple - 1) * 100, 2),
        },
        "tickers": compute_per_ticker_stats(trades, n_tickers),
    }

    out_path = config.RESULTS_DIR / "training_backtest.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path} (dashboard will pick this up automatically)")


if __name__ == "__main__":
    main()
