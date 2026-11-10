"""
Simulates a simple trading strategy built on the model's predictions, using
the same walk-forward folds as training so results stay honest (no lookahead).

Strategy (deliberately simple -- a starting point, not a recommendation):
  - If the model predicts "up" with probability > CONFIDENCE_THRESHOLD,
    enter at the next session's open and exit at the open after the
    configured trading-day horizon.
  - Otherwise, stay in cash.
  - Each trade pays TRANSACTION_COST_BPS (round-trip) to approximate real
    trading costs -- a model can look accurate and still lose money once
    this is included.

This does not model broker-specific fees/slippage, taxes, partial fills,
or market impact. Treat it as a sanity check, not a real P&L estimate.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import walk_forward_splits, load_features

CONFIDENCE_THRESHOLD = 0.55
TRANSACTION_COST_BPS = config.ROUND_TRIP_COST_BPS


def simulate_fold(train_df, test_df):
    from lightgbm import LGBMClassifier
    from models.train_model import FEATURE_COLUMNS

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, verbose=-1)
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    test_df["future_return"] = test_df["forward_return"]
    cost = TRANSACTION_COST_BPS / 10_000
    test_df["net_return"] = test_df["future_return"] - cost
    return test_df[
        ["ticker", "pred_prob", "future_return", "net_return", "label", "forward_exit_date"]
    ]


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
    if signals.empty:
        return signals.copy()
    signals = signals.sort_index()
    kept_rows = []

    for ticker, group in signals.groupby("ticker"):
        next_allowed_date = None
        for date, row in group.iterrows():
            if next_allowed_date is not None and date < next_allowed_date:
                continue  # still "in a trade" on this ticker, skip
            kept_rows.append(row)
            next_allowed_date = pd.Timestamp(row["forward_exit_date"])

    return pd.DataFrame(kept_rows)


def compute_equity_curve(trades: pd.DataFrame, n_tickers: int) -> float:
    """
    Compounds returns assuming capital is split evenly across tickers
    up front (e.g. 5 tickers -> 20% of capital each), and each ticker's
    slice compounds independently through its own non-overlapping trades.
    Returns the final portfolio multiple (e.g. 1.35 = +35% total).
    """
    starting_share = 1.0 / n_tickers
    final_value = starting_share * (n_tickers - trades["ticker"].nunique())

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
    trades_per_ticker = trades.groupby("ticker").size().to_dict()
    if not trades_per_ticker:
        return 1.0

    candidates = {}
    for ticker, n_trades_needed in trades_per_ticker.items():
        ticker_df = df[df["ticker"] == ticker].sort_index()
        return_column = (
            "forward_return" if "forward_return" in ticker_df else "future_return"
        )
        ticker_df = ticker_df.dropna(subset=[return_column])
        candidates[ticker] = (
            n_trades_needed,
            ticker_df.index.to_numpy(),
            pd.to_datetime(ticker_df["forward_exit_date"]).to_numpy(),
            ticker_df[return_column].to_numpy(dtype=float),
        )

    run_results = []
    for _ in range(n_runs):
        starting_share = 1.0 / n_tickers
        final_value = starting_share * (n_tickers - len(trades_per_ticker))

        for n_trades_needed, dates, exit_dates, returns in candidates.values():
            equity = starting_share
            taken = 0
            next_allowed_date = None
            shuffled_starts = rng.permutation(len(dates))
            for start_idx in shuffled_starts:
                if taken >= n_trades_needed:
                    break
                start_date = dates[start_idx]
                if next_allowed_date is not None and start_date < next_allowed_date:
                    continue
                trade_return = returns[start_idx] - cost
                equity *= (1 + trade_return)
                next_allowed_date = exit_dates[start_idx]
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
    starting_share = 1.0 / len(tickers)
    final_value = 1.0

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker]
        ticker_df = ticker_df[ticker_df.index >= earliest_test_date]
        if len(ticker_df) < 2:
            continue
        start_price = ticker_df["close"].iloc[0]
        end_price = ticker_df["close"].iloc[-1]
        final_value += starting_share * (end_price / start_price - 1)

    return final_value


def compute_period_benchmark(
    df: pd.DataFrame, tickers: list, start_date, end_date
) -> float:
    """Equal-weight buy-and-hold return over one specific validation fold."""
    starting_share = 1.0 / len(tickers)
    final_value = 1.0
    for ticker in tickers:
        ticker_df = df[
            (df["ticker"] == ticker)
            & (df.index >= start_date)
            & (df.index <= end_date)
        ]
        if len(ticker_df) >= 2:
            final_value += starting_share * (
                ticker_df["close"].iloc[-1] / ticker_df["close"].iloc[0] - 1
            )
    return final_value


def compute_portfolio_equity_curve(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    tickers: list,
    start_date,
    end_date,
    cost_bps: int = TRANSACTION_COST_BPS,
) -> pd.Series:
    """Mark equal-weight ticker sleeves to market at each available open."""
    calendar = df.index[(df.index >= start_date) & (df.index <= end_date)]
    calendar = calendar.unique().sort_values()
    if calendar.empty:
        raise ValueError("No market bars available in the requested evaluation period.")

    cost = cost_bps / 10_000
    sleeve_curves = []
    starting_share = 1 / len(tickers)

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].sort_index()
        prices = ticker_df["open"].dropna()
        daily_returns = pd.Series(0.0, index=calendar)
        ticker_signals = signals[signals["ticker"] == ticker].sort_index()
        next_allowed_date = None

        for signal_date, signal in ticker_signals.iterrows():
            if next_allowed_date is not None and signal_date < next_allowed_date:
                continue
            entry_pos = prices.index.searchsorted(signal_date, side="right")
            exit_date = pd.Timestamp(signal["forward_exit_date"])
            exit_pos = prices.index.get_indexer([exit_date])[0]
            if entry_pos >= len(prices) or exit_pos <= entry_pos:
                continue

            path_dates = prices.index[entry_pos:exit_pos + 1]
            if path_dates[0] not in daily_returns.index or path_dates[-1] not in daily_returns.index:
                continue

            daily_returns.loc[path_dates[0]] -= cost / 2
            for position in range(entry_pos + 1, exit_pos + 1):
                date = prices.index[position]
                daily_return = prices.iloc[position] / prices.iloc[position - 1] - 1
                if position == exit_pos:
                    daily_return = (1 + daily_return) * (1 - cost / 2) - 1
                daily_returns.loc[date] += daily_return
            next_allowed_date = exit_date

        sleeve_curves.append(
            starting_share * (1 + daily_returns).cumprod().rename(ticker)
        )

    return pd.concat(sleeve_curves, axis=1).sum(axis=1).rename("equity")


def compute_buy_and_hold_equity_curve(
    df: pd.DataFrame, tickers: list, start_date, end_date
) -> pd.Series:
    calendar = df.index[(df.index >= start_date) & (df.index <= end_date)]
    calendar = calendar.unique().sort_values()
    if calendar.empty:
        raise ValueError("No market bars available in the requested benchmark period.")

    share = 1 / len(tickers)
    sleeves = []
    for ticker in tickers:
        prices = df.loc[df["ticker"] == ticker, "open"].dropna()
        prices = prices[prices.index <= end_date]
        if prices.empty:
            sleeves.append(pd.Series(share, index=calendar, name=ticker))
            continue
        after_signal = prices[prices.index > start_date]
        start_price = after_signal.iloc[0] if not after_signal.empty else prices.iloc[-1]
        daily_prices = prices.reindex(calendar).ffill().fillna(start_price)
        sleeves.append((share * daily_prices / start_price).rename(ticker))
    return pd.concat(sleeves, axis=1).sum(axis=1).rename("equity")


def risk_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    daily_returns = equity.pct_change().dropna()
    volatility = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = (
        float(daily_returns.mean() / volatility * np.sqrt(252))
        if volatility > 0 else None
    )
    drawdown = equity / equity.cummax() - 1
    gains = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()
    return {
        "maxDrawdown": round(float(drawdown.min()) * 100, 2),
        "annualizedVolatility": round(volatility * np.sqrt(252) * 100, 2),
        "annualizedSharpe": round(sharpe, 3) if sharpe is not None else None,
        "profitFactor": round(float(gains / losses), 3) if losses > 0 else None,
        "largestLossPerTrade": round(
            float(trades["net_return"].min()) * 100, 2
        ) if not trades.empty else None,
    }


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
        print("No walk-forward predictions were produced; check feature history.")
        return

    oof_predictions = pd.concat(all_signals)
    raw_signals = oof_predictions[oof_predictions["pred_prob"] > CONFIDENCE_THRESHOLD]
    trades = enforce_non_overlapping(raw_signals)

    win_rate = (trades["net_return"] > 0).mean() if not trades.empty else 0.0
    avg_return = trades["net_return"].mean() if not trades.empty else 0.0
    total_trades = len(trades)
    n_tickers = len(config.TICKERS)
    evaluation_start = oof_predictions.index.min()
    evaluation_end = max(
        oof_predictions.index.max(),
        pd.to_datetime(trades["forward_exit_date"]).max()
        if not trades.empty else oof_predictions.index.max(),
    )
    portfolio_equity = compute_portfolio_equity_curve(
        df, trades, config.TICKERS, evaluation_start, evaluation_end
    )
    benchmark_equity = compute_buy_and_hold_equity_curve(
        df, config.TICKERS, evaluation_start, evaluation_end
    )
    portfolio_multiple = float(portfolio_equity.iloc[-1])
    benchmark_multiple = float(benchmark_equity.iloc[-1])
    portfolio_risk = risk_metrics(portfolio_equity, trades)
    period_metrics = []
    for fold_i, fold_predictions in oof_predictions.groupby("fold"):
        fold_trades = enforce_non_overlapping(
            fold_predictions[fold_predictions["pred_prob"] > CONFIDENCE_THRESHOLD]
        )
        fold_equity = compute_portfolio_equity_curve(
            df, fold_trades, config.TICKERS,
            fold_predictions.index.min(),
            max(
                fold_predictions.index.max(),
                pd.to_datetime(fold_trades["forward_exit_date"]).max()
                if not fold_trades.empty else fold_predictions.index.max(),
            ),
        )
        fold_benchmark_equity = compute_buy_and_hold_equity_curve(
            df, config.TICKERS, fold_predictions.index.min(),
            max(
                fold_predictions.index.max(),
                pd.to_datetime(fold_trades["forward_exit_date"]).max()
                if not fold_trades.empty else fold_predictions.index.max(),
            ),
        )
        fold_multiple = float(fold_equity.iloc[-1])
        fold_benchmark = float(fold_benchmark_equity.iloc[-1])
        period_metrics.append({
            "fold": int(fold_i),
            "start": fold_predictions.index.min().strftime("%Y-%m-%d"),
            "end": fold_predictions.index.max().strftime("%Y-%m-%d"),
            "rows": int(len(fold_predictions)),
            "trades": int(len(fold_trades)),
            "modelReturn": round((fold_multiple - 1) * 100, 2),
            "benchmarkReturn": round((fold_benchmark - 1) * 100, 2),
            "brierScore": round(
                float(brier_score_loss(
                    fold_predictions["label"], fold_predictions["pred_prob"]
                )),
                4,
            ),
        })
        print(
            f"Fold {fold_i} ({period_metrics[-1]['start']} to "
            f"{period_metrics[-1]['end']}): "
            f"{len(fold_trades)} trades, model "
            f"{period_metrics[-1]['modelReturn']:.2f}%, buy-and-hold "
            f"{period_metrics[-1]['benchmarkReturn']:.2f}%"
        )

    print(f"Raw signals fired:            {len(raw_signals)}")
    print(f"Non-overlapping trades taken:  {total_trades}  "
          f"(one position per ticker at a time)")
    print(f"Win rate (after costs):        {win_rate:.1%}")
    print(f"Avg return per trade:          {avg_return:.2%}")
    print(f"Compounded portfolio return:    {(portfolio_multiple - 1):.2%}  "
          f"(capital split evenly across {n_tickers} tickers)")
    print(
        "Risk (mark-to-market):          "
        f"max drawdown {portfolio_risk['maxDrawdown']}%, "
        f"annualized vol {portfolio_risk['annualizedVolatility']}%, "
        f"Sharpe {portfolio_risk['annualizedSharpe']}"
    )
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

    if not trades.empty:
        print_per_ticker_breakdown(trades, n_tickers)

    print("\nRunning random-entry baseline (same trade counts, random timing, 20 runs)...")
    random_multiple = simulate_random_baseline(oof_predictions, trades, n_tickers)
    print(f"Random-entry avg return:        {(random_multiple - 1):.2%}  "
          f"(vs. buy-and-hold {(benchmark_multiple - 1):.2%})")

    random_edge = portfolio_multiple - random_multiple
    if random_edge > 0 and edge > 0:
        print(
            f"\nModel exceeded random timing by {random_edge:.1%} and buy-and-hold; "
            "this is promising, but still not proof of a durable edge."
        )
    elif random_edge > 0:
        print(
            f"\nModel exceeded random timing by {random_edge:.1%}, but still "
            f"underperformed buy-and-hold by {abs(edge):.1%}."
        )
    else:
        print(
            f"\nModel underperformed random timing by {abs(random_edge):.1%}; "
            "the backtest does not support using this strategy."
        )

    print(f"\nAssumed round-trip fees + slippage: {TRANSACTION_COST_BPS} bps")
    print("Still simplified -- no taxes, partial fills, or market impact.")
    print_cost_sensitivity(trades, n_tickers)

    save_results_json(trades, n_tickers, total_trades, win_rate, avg_return,
                       portfolio_multiple, benchmark_multiple, random_multiple,
                       oof_predictions, period_metrics, portfolio_risk)


def compute_cost_sensitivity(trades: pd.DataFrame, n_tickers: int) -> list:
    rows = []
    for cost_bps in (10, TRANSACTION_COST_BPS, 50):
        adjusted = trades.copy()
        adjusted["net_return"] = (
            adjusted["future_return"] - cost_bps / 10_000
        )
        multiple = compute_equity_curve(adjusted, n_tickers)
        rows.append({
            "roundTripCostBps": cost_bps,
            "portfolioReturn": round((multiple - 1) * 100, 2),
        })
    return rows


def print_cost_sensitivity(trades: pd.DataFrame, n_tickers: int) -> None:
    print("\nCost sensitivity (same trades, different round-trip assumptions):")
    for row in compute_cost_sensitivity(trades, n_tickers):
        print(f"  {row['roundTripCostBps']:>2} bps: {row['portfolioReturn']:.2f}%")


def save_results_json(trades, n_tickers, total_trades, win_rate, avg_return,
                       portfolio_multiple, benchmark_multiple, random_multiple,
                       oof_predictions, period_metrics, portfolio_risk):
    """
    Writes results/training_backtest.json -- a snapshot the dashboard
    server reads and serves. Overwritten every time this script runs, so
    the dashboard always reflects your most recent backtest.
    """
    import json
    from datetime import datetime

    from models.train_model import calibration_holdout_summary, calibration_summary

    output = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "trades": total_trades,
            "winRate": round(float(win_rate) * 100, 1),
            "avgReturnPerTrade": round(float(avg_return) * 100, 2),
            "modelReturn": round((portfolio_multiple - 1) * 100, 2),
            "benchmarkReturn": round((benchmark_multiple - 1) * 100, 2),
            "randomReturn": round((random_multiple - 1) * 100, 2),
            "risk": portfolio_risk,
        },
        "tickers": compute_per_ticker_stats(trades, n_tickers),
        "assumedRoundTripCostBps": TRANSACTION_COST_BPS,
        "costSensitivity": compute_cost_sensitivity(trades, n_tickers),
        "periodMetrics": period_metrics,
        "calibration": calibration_summary(
            oof_predictions["label"], oof_predictions["pred_prob"]
        ),
        "calibrationHoldout": calibration_holdout_summary(oof_predictions),
    }

    out_path = config.RESULTS_DIR / "training_backtest.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path} (dashboard will pick this up automatically)")


if __name__ == "__main__":
    main()
