import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from openai import OpenAI

import streamlit as st
import os
from openai import OpenAI

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Yahoo headers (reduce 429)
# ----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ----------------------------
# Yahoo search: company -> candidates
# ----------------------------
def search_company(name: str, retries: int = 3):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": name, "quotesCount": 7, "newsCount": 0}

    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json().get("quotes", [])
        if r.status_code == 429:
            wait = 2 * (attempt + 1)
            time.sleep(wait)
            continue
        r.raise_for_status()

    raise Exception("Yahoo rate limit exceeded. Try again in a minute.")

# ----------------------------
# OpenAI: pick best ticker
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
    resp = client.responses.create(model="gpt-4.1", input=prompt)
    return resp.output_text

def extract_ticker(ai_text: str) -> str | None:
    for line in ai_text.splitlines():
        if line.strip().upper().startswith("TICKER:"):
            return line.split(":", 1)[1].strip()
    return None

# ----------------------------
# Yahoo chart: ticker -> 1y daily closes
# ----------------------------
def yahoo_price_history(symbol: str, range_: str = "1y", interval: str = "1d", retries: int = 3):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}

    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 429:
            wait = 2 * (attempt + 1)
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
            raise Exception(f"Not enough price data for {symbol}.")
        return df

    raise Exception("Yahoo chart rate limit exceeded. Try again shortly.")

# ----------------------------
# Metrics
# ----------------------------
def compute_trend_metrics(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()

    start_price = float(d["close"].iloc[0])
    end_price = float(d["close"].iloc[-1])
    total_return = (end_price / start_price) - 1.0

    vol = float(d["ret"].std() * math.sqrt(252))

    running_max = d["close"].cummax()
    drawdown = (d["close"] / running_max) - 1.0
    max_dd = float(drawdown.min())

    x = np.arange(len(d))
    y = np.log(d["close"].values)
    slope, _ = np.polyfit(x, y, 1)
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
# Chart (matplotlib) -> Streamlit
# ----------------------------
def make_chart(df: pd.DataFrame, symbol: str):
    d = df.copy()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()

    fig = plt.figure()
    plt.plot(d["date"], d["close"], label="Close")
    plt.plot(d["date"], d["ma20"], label="MA20")
    plt.plot(d["date"], d["ma50"], label="MA50")
    plt.title(f"{symbol} Price Trend (1y)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    return fig

# ----------------------------
# AI explanation
# ----------------------------
def explain_trend(symbol: str, metrics: dict):
    prompt = f"""
You are a helpful financial trend analyst.

Given these computed metrics for {symbol} (1-year daily data):
{json.dumps(metrics, indent=2)}

Write a clear explanation for a non-expert:
- Is it overall uptrend / downtrend / sideways? (justify using return + growth estimate + MA signal)
- Mention total return, volatility, max drawdown in plain language
- End with a short disclaimer: not financial advice

Keep it concise: ~8-12 sentences.
"""
    resp = client.responses.create(model="gpt-4.1", input=prompt)
    return resp.output_text

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Finance Trend Agent", layout="wide")
st.title("AI Finance Trend Agent")
st.write("Enter a company name. The agent validates the ticker, pulls 1-year prices, charts the trend, and explains it.")

company = st.text_input("Company name", placeholder="Example: Apple, Tesla, Coca Cola, Microsoft")

col1, col2 = st.columns([1, 1])
with col1:
    range_choice = st.selectbox("Time range", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
with col2:
    interval_choice = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

if st.button("Analyze"):
    if not company.strip():
        st.error("Please enter a company name.")
        st.stop()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Set it in your terminal/environment and restart Streamlit.")
        st.stop()
    if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets.")
    st.stop()


    try:
        with st.spinner("Searching Yahoo Finance..."):
            candidates = search_company(company.strip())

        with st.spinner("Validating ticker with AI..."):
            ai_validation = analyze_company(company.strip(), candidates)
            ticker = extract_ticker(ai_validation)

        st.subheader("AI Validation")
        st.code(ai_validation)

        if not ticker:
            st.warning("AI couldn't confidently pick a ticker. Try a more specific company name.")
            st.stop()

        with st.spinner(f"Fetching price history for {ticker}..."):
            df = yahoo_price_history(ticker, range_=range_choice, interval=interval_choice)

        with st.spinner("Computing metrics..."):
            metrics = compute_trend_metrics(df)

        with st.spinner("Generating chart..."):
            fig = make_chart(df, ticker)

        st.subheader("Trend Chart")
        st.pyplot(fig)

        st.subheader("Computed Metrics")
        st.json(metrics)

        with st.spinner("Writing AI trend explanation..."):
            explanation = explain_trend(ticker, metrics)

        st.subheader("AI Trend Explanation")
        st.write(explanation)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("If you see a Yahoo 429 error, wait 1–3 minutes and try again.")
