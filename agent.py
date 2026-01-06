import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from openai import OpenAI

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Yahoo Finance: use browser-like headers to reduce 429 blocks
# ----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ----------------------------
# 1) Search company name -> candidates (Yahoo search)
# ----------------------------
def search_company(name: str, retries: int = 3):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": name, "quotesCount": 7, "newsCount": 0}

    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)

        if r.status_code == 200:
            data = r.json()
            return data.get("quotes", [])

        if r.status_code == 429:
            wait = 2 * (attempt + 1)
            print(f"⚠ Yahoo rate limit hit (429). Waiting {wait}s...")
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise Exception("Yahoo Finance rate limit exceeded. Try again later.")

# ----------------------------
# 2) Use OpenAI to choose best ticker from candidates
# ----------------------------
def analyze_company(name: str, candidates: list[dict]):
    prompt = f"""
User entered company name: {name}

Yahoo Finance candidates (JSON):
{json.dumps(candidates, indent=2)}

Task:
- Pick the best matching STOCK ticker symbol (prefer US equities if available).
- Ignore ETFs/ETNs/funds unless user clearly typed an ETF name.
- If ambiguous, ask ONE short clarifying question.

If clear, output exactly:
TICKER: <symbol>
COMPANY: <company name>
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    return resp.output_text

def extract_ticker(ai_text: str) -> str | None:
    # simple parser: look for a line like "TICKER: KO"
    for line in ai_text.splitlines():
        if line.strip().upper().startswith("TICKER:"):
            return line.split(":", 1)[1].strip()
    return None

# ----------------------------
# 3) Fetch price history from Yahoo chart endpoint
# ----------------------------
def yahoo_price_history(symbol: str, range_: str = "1y", interval: str = "1d", retries: int = 3):
    """
    Returns a DataFrame with columns: date, close
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}

    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)

        if r.status_code == 429:
            wait = 2 * (attempt + 1)
            print(f"⚠ Yahoo rate limit hit (429) on chart. Waiting {wait}s...")
            time.sleep(wait)
            continue

        r.raise_for_status()
        raw = r.json()

        result = (raw.get("chart", {}) or {}).get("result", [])
        if not result:
            raise Exception(f"No chart data returned for {symbol}")

        res0 = result[0]
        timestamps = res0.get("timestamp", [])
        quote = ((res0.get("indicators", {}) or {}).get("quote", []) or [{}])[0]
        closes = quote.get("close", [])

        rows = []
        for ts, c in zip(timestamps, closes):
            if c is None:
                continue
            dt = datetime.utcfromtimestamp(ts).date()
            rows.append({"date": dt, "close": float(c)})

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if df.empty or len(df) < 30:
            raise Exception(f"Not enough price data for {symbol} to analyze.")
        return df

    raise Exception("Yahoo Finance rate limit exceeded on chart. Try again later.")

# ----------------------------
# 4) Compute trend metrics
# ----------------------------
def compute_trend_metrics(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()

    start_price = float(d["close"].iloc[0])
    end_price = float(d["close"].iloc[-1])
    total_return = (end_price / start_price) - 1.0

    # annualized volatility (daily)
    vol = float(d["ret"].std() * math.sqrt(252))

    # max drawdown
    running_max = d["close"].cummax()
    drawdown = (d["close"] / running_max) - 1.0
    max_dd = float(drawdown.min())

    # Trend slope: linear regression on log(price)
    x = np.arange(len(d))
    y = np.log(d["close"].values)
    slope, _ = np.polyfit(x, y, 1)

    # approximate annualized growth from slope
    annualized_growth = float(math.exp(slope * 252) - 1.0)

    ma_signal = None
    last_ma20 = d["ma20"].iloc[-1]
    last_ma50 = d["ma50"].iloc[-1]
    if not np.isnan(last_ma20) and not np.isnan(last_ma50):
        ma_signal = "bullish (MA20 > MA50)" if last_ma20 > last_ma50 else "bearish (MA20 < MA50)"

    return {
        "points": int(len(d)),
        "start_date": str(d["date"].iloc[0]),
        "end_date": str(d["date"].iloc[-1]),
        "start_price": start_price,
        "end_price": end_price,
        "total_return_pct": float(total_return * 100),
        "annualized_volatility_pct": float(vol * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "annualized_growth_estimate_pct": float(annualized_growth * 100),
        "ma_signal": ma_signal
    }

# ----------------------------
# 5) Save chart to PNG
# ----------------------------
def save_price_chart(df: pd.DataFrame, symbol: str, out_path: str = "trend.png"):
    d = df.copy()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()

    plt.figure()
    plt.plot(d["date"], d["close"], label="Close")
    plt.plot(d["date"], d["ma20"], label="MA20")
    plt.plot(d["date"], d["ma50"], label="MA50")
    plt.title(f"{symbol} Price Trend (1y)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path

# ----------------------------
# 6) Ask OpenAI to explain trend using metrics
# ----------------------------
def explain_trend(symbol: str, metrics: dict):
    prompt = f"""
You are a helpful financial trend analyst.

Given these computed metrics for {symbol} (1-year daily data):
{json.dumps(metrics, indent=2)}

Write a clear explanation for a non-expert:
- Is it overall uptrend / downtrend / sideways? (justify using return + slope + MA signal)
- Mention total return, volatility, max drawdown in plain language
- Mention MA20 vs MA50 signal if present
- End with a short disclaimer: not financial advice

Keep it concise: ~8-12 sentences.
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    return resp.output_text

# ----------------------------
# Main app flow
# ----------------------------
if __name__ == "__main__":
    company = input("Enter company name (e.g., Apple, Tesla, Coca Cola): ").strip()
    candidates = search_company(company)

    print("\nYahoo Candidates (raw):")
    print(candidates)

    print("\nAI Validation:")
    ai_text = analyze_company(company, candidates)
    print(ai_text)

    ticker = extract_ticker(ai_text)
    if not ticker:
        print("\nThe AI could not confidently pick a ticker. Please re-run with a more specific name.")
        raise SystemExit(1)

    # Fetch prices + analyze
    print(f"\nFetching 1-year price history for: {ticker}")
    df = yahoo_price_history(ticker, range_="1y", interval="1d")

    metrics = compute_trend_metrics(df)
    print("\nComputed Metrics:")
    print(json.dumps(metrics, indent=2))

    chart_path = save_price_chart(df, ticker, out_path=f"{ticker}_trend.png")
    print(f"\nSaved chart: {chart_path}")

    # AI explanation
    print("\nAI Trend Explanation:\n")
    explanation = explain_trend(ticker, metrics)
    print(explanation)
