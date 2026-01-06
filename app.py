# app.py
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from openai import OpenAI


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AI Finance Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Finance Agent")
st.caption("Search a company → find ticker → pull 1Y price trend → show chart + metrics + AI explanation.")


# -------------------------
# Helpers: API key (Cloud + local)
# -------------------------
def get_openai_api_key() -> Optional[str]:
    # Streamlit Cloud: st.secrets
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        key = str(st.secrets["OPENAI_API_KEY"]).strip()
        return key if key else None

    # Local dev: environment variable
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key if key else None


OPENAI_API_KEY = get_openai_api_key()

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets (Settings → Secrets).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------
# Yahoo Finance request setup
# -------------------------
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def yahoo_get(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5) -> requests.Response:
    """
    GET with retry/backoff for Yahoo rate limiting (429) or transient failures.
    """
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)

            if r.status_code == 200:
                return r

            # 429 rate limit: wait a bit longer each time
            if r.status_code == 429:
                wait_s = 2 * (attempt + 1)
                time.sleep(wait_s)
                continue

            # Other errors: raise
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Yahoo request failed after {retries} tries: {last_err}")


# -------------------------
# Yahoo Search (company -> candidates)
# -------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def search_company(company_name: str) -> List[Dict[str, Any]]:
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 8, "newsCount": 0}
    r = yahoo_get(url, params=params)
    data = r.json()
    return data.get("quotes", []) or []


# -------------------------
# AI: pick best ticker
# -------------------------
def pick_ticker_with_ai(user_input: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns dict: { "ticker": "...", "company": "...", "reason": "..."}
    """
    # Keep candidate payload small & relevant
    slim = []
    for c in candidates[:8]:
        slim.append({
            "symbol": c.get("symbol"),
            "shortname": c.get("shortname"),
            "longname": c.get("longname"),
            "exchDisp": c.get("exchDisp"),
            "typeDisp": c.get("typeDisp"),
        })

    prompt = f"""
You are helping match a user's company input to the most likely Yahoo Finance ticker.

User input: {user_input}

Candidates (from Yahoo):
{json.dumps(slim, indent=2)}

Rules:
- Pick the best match for a publicly traded company if available.
- If user input is ambiguous, still pick the best candidate, but explain the ambiguity.
- Output MUST be valid JSON only.

Output JSON schema:
{{
  "ticker": "string",
  "company": "string",
  "reason": "string"
}}
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )

    text = (resp.output_text or "").strip()
    # Defensive JSON parse
    try:
        return json.loads(text)
    except Exception:
        # fallback if model returns extra text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


# -------------------------
# Yahoo price history (ticker -> DataFrame)
# -------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def yahoo_price_history(ticker: str, range_: str = "1y", interval: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": interval}
    r = yahoo_get(url, params=params)
    data = r.json()

    chart = data.get("chart", {})
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError("Yahoo chart API returned no result.")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {}).get("quote", [])
    if not indicators:
        raise RuntimeError("Yahoo chart API missing quote indicators.")

    quote0 = indicators[0]
    closes = quote0.get("close") or []
    opens = quote0.get("open") or []
    highs = quote0.get("high") or []
    lows = quote0.get("low") or []
    volumes = quote0.get("volume") or []

    # Build DF
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Clean
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# -------------------------
# Trend metrics
# -------------------------
def compute_trend_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)

    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    n = len(close)

    # Daily returns
    ret = close.pct_change().dropna()
    if len(ret) < 2:
        raise RuntimeError("Not enough price points to compute returns.")

    # Annualization constants
    TRADING_DAYS = 252.0

    total_return = (end / start) - 1.0
    annualized_return = (end / start) ** (TRADING_DAYS / max(n - 1, 1)) - 1.0
    annualized_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # Max drawdown
    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    # Moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # Simple crossover signal
    signal = None
    if not np.isnan(ma20.iloc[-1]) and not np.isnan(ma50.iloc[-1]):
        signal = "bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "bearish"

    return {
        "start_price": round(start, 4),
        "end_price": round(end, 4),
        "days": int(n),
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct_est": round(annualized_return * 100, 2),
        "annualized_volatility_pct": round(annualized_vol * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "ma20_last": None if np.isnan(ma20.iloc[-1]) else round(float(ma20.iloc[-1]), 4),
        "ma50_last": None if np.isnan(ma50.iloc[-1]) else round(float(ma50.iloc[-1]), 4),
        "ma20_vs_ma50_signal": signal,
    }


def make_chart(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["close"], label="Close")
    ax.set_title(f"{ticker} — 1Y Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


# -------------------------
# AI: trend explanation
# -------------------------
def explain_trend(ticker: str, company: str, metrics: Dict[str, Any]) -> str:
    prompt = f"""
You are a finance assistant. Explain the 1-year trend using the provided metrics.
Keep it clear and beginner-friendly. Mention risk (volatility/drawdown), trend direction, and what the MA signal implies.
Do NOT give financial advice. End with a short disclaimer.

Ticker: {ticker}
Company: {company}

Metrics:
{json.dumps(metrics, indent=2)}

Output format:
- 1 short paragraph (4-6 sentences)
- 3 bullet points: "Trend", "Risk", "Notes"
- Final line: "Disclaimer: ..."

"""
    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )
    return (resp.output_text or "").strip()


# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("Inputs")
    company = st.text_input("Enter company name (e.g., Coca Cola, Tesla, Google)", value="")
    range_opt = st.selectbox("Price range", ["6mo", "1y", "2y", "5y"], index=1)
    interval_opt = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.divider()
    st.caption("If Yahoo rate-limits you (429), wait 1–3 minutes and try again.")


if not company.strip():
    st.info("Type a company name in the left sidebar to begin.")
    st.stop()


# -------------------------
# Run workflow
# -------------------------
try:
    with st.spinner("Searching Yahoo Finance…"):
        candidates = search_company(company.strip())

    if not candidates:
        st.error("No Yahoo Finance matches found. Try a more specific company name.")
        st.stop()

    # Show candidates
    with st.expander("Yahoo candidates (debug)", expanded=False):
        st.json(candidates)

    with st.spinner("Validating best ticker using AI…"):
        picked = pick_ticker_with_ai(company.strip(), candidates)

    ticker = (picked.get("ticker") or "").strip()
    company_name = (picked.get("company") or company.strip()).strip()
    reason = (picked.get("reason") or "").strip()

    if not ticker:
        st.error("AI could not determine a ticker. Try a more specific company name.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Selected")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Company:** {company_name}")
    with col2:
        st.subheader("Why this match?")
        st.write(reason if reason else "—")

    with st.spinner(f"Fetching {range_opt} price history for {ticker}…"):
        df = yahoo_price_history(ticker, range_=range_opt, interval=interval_opt)

    metrics = compute_trend_metrics(df)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Trend Chart")
        fig = make_chart(df, ticker)
        st.pyplot(fig)

    with right:
        st.subheader("Computed Metrics")
        st.json(metrics)

    with st.spinner("Writing AI trend explanation…"):
        explanation = explain_trend(ticker, company_name, metrics)

    st.subheader("AI Trend Explanation")
    st.write(explanation)

except requests.exceptions.HTTPError as e:
    st.error(f"HTTP error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
