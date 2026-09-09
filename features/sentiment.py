"""
Shared sentiment utilities used by both the historical fetch script
(data/fetch_news_sentiment.py) and live prediction (serve/live_predict.py).

Two separate things happen here:
  1. FETCHING news text -- uses Finnhub's free "company news" endpoint,
     which (unlike most free news APIs) supports historical date ranges,
     not just the last month.
  2. SCORING that text -- uses VADER, a rule-based sentiment analyzer.
     No training, no API key, and no internet access needed for scoring
     itself -- it's a bundled lexicon, so it works offline once installed.

Getting a free Finnhub API key:
  1. Sign up at https://finnhub.io/register (free tier)
  2. Copy your API key from the dashboard
  3. Set it as an environment variable before running the fetch scripts:
       Windows (PowerShell):  $env:FINNHUB_API_KEY="your_key_here"
       Mac/Linux (bash):      export FINNHUB_API_KEY="your_key_here"

If no key is set, every function here degrades gracefully to "no
articles found" / a sentiment score of 0 -- the rest of the pipeline
still runs, just without this signal. This is deliberate: sentiment is
an optional enhancement, not a required dependency.

HONEST LIMITATION: Finnhub's free tier is best-effort for historical
coverage -- some tickers or older dates may return few or no articles.
Check the printed article counts when you run fetch_news_sentiment.py; a
ticker with very few articles won't get much signal from this feature.
"""

import time

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """
    Returns VADER's "compound" sentiment score: roughly -1 (very negative)
    to +1 (very positive), 0 for neutral or empty text.
    """
    if not text:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def fetch_company_news(ticker: str, from_date: str, to_date: str,
                        api_key: str, max_retries: int = 3) -> list:
    """
    from_date / to_date: "YYYY-MM-DD" strings.
    Returns a list of article dicts (each with 'headline', 'summary',
    'datetime' as a unix timestamp), or [] on any failure -- missing key,
    rate limit, network error, or no news found. Callers should treat an
    empty list as "no sentiment signal available," not as an error to
    surface to the end user.
    """
    if not api_key:
        return []

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": api_key}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:  # rate limited -- back off and retry
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  news fetch failed for {ticker}: {e}")
                return []
            time.sleep(1)
    return []


def aggregate_daily_sentiment(articles: list) -> pd.DataFrame:
    """
    Groups articles by calendar day. Returns a DataFrame indexed by date
    (as a Timestamp) with columns:
      - news_sentiment: average VADER compound score across that day's
        headlines + summaries
      - news_volume: number of articles that day (a rough proxy for how
        much attention the stock is getting)
    Returns an empty DataFrame (same columns, no rows) if given no
    articles -- callers should treat this as "0 for every date."
    """
    empty = pd.DataFrame(columns=["news_sentiment", "news_volume"])
    if not articles:
        return empty

    rows = []
    for a in articles:
        ts = a.get("datetime")
        if not ts:
            continue
        date = pd.to_datetime(ts, unit="s").normalize()
        text = f"{a.get('headline', '')} {a.get('summary', '')}".strip()
        rows.append({"date": date, "score": score_text(text)})

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    daily = df.groupby("date")["score"].agg(["mean", "count"])
    daily.columns = ["news_sentiment", "news_volume"]
    return daily
