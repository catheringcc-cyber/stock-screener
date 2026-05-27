"""Stock universe loaders.

Default universe is S&P 500 ∪ Nasdaq 100 (~600 names), fetched from Wikipedia.
This gives the most relevant momentum-driven large caps without the slow movers
that dominate the bottom half of Russell 1000. Falls back to a hardcoded
short list of large-cap momentum names if Wikipedia is unreachable.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pandas as pd
import requests

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)


def _read_html_with_ua(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ARM", "AMD", "PLTR", "NFLX", "ORCL", "CRM", "ADBE", "QCOM", "INTU", "AMAT",
    "MU", "LRCX", "KLAC", "MRVL", "PANW", "CRWD", "SNOW", "DDOG", "NET", "MDB",
    "SHOP", "UBER", "ABNB", "COIN", "MSTR", "HOOD", "SOFI", "AFRM", "SQ",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "SPGI", "V", "MA", "AXP",
    "BRK-B", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "DHR", "ABT",
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO",
    "HD", "LOW", "WMT", "COST", "TGT", "NKE", "MCD", "SBUX", "DIS", "CMCSA",
    "BA", "CAT", "DE", "GE", "HON", "RTX", "LMT", "NOC", "GD",
    "TSM", "ASML", "NVO", "ASTS", "RKLB", "JOBY", "SMCI", "VRT", "ANET", "DELL",
]


def _fetch_sp500() -> List[str]:
    tables = _read_html_with_ua(SP500_URL)
    df = tables[0]
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def _fetch_ndx() -> List[str]:
    tables = _read_html_with_ua(NDX_URL)
    # The constituents table column is "Ticker" or "Symbol" depending on Wikipedia rev
    for df in tables:
        for col in ("Ticker", "Symbol"):
            if col in df.columns and len(df) >= 90:  # NDX is 100 names
                return df[col].astype(str).str.replace(".", "-", regex=False).tolist()
    raise ValueError("Could not locate Nasdaq-100 constituents table")


def get_universe(name: str = "sp500_ndx") -> List[str]:
    """Return list of tickers. name: sp500_ndx | sp500 | fallback."""
    cache_file = CACHE_DIR / f"universe_{name}.csv"
    if cache_file.exists():
        age_days = (pd.Timestamp.utcnow() - pd.Timestamp(cache_file.stat().st_mtime, unit="s", tz="UTC")).days
        if age_days < 7:
            return pd.read_csv(cache_file)["ticker"].tolist()

    tickers: List[str] = []
    try:
        if name == "sp500_ndx":
            sp = _fetch_sp500()
            try:
                ndx = _fetch_ndx()
            except Exception:
                ndx = []
            tickers = sorted(set(sp) | set(ndx))
        elif name == "sp500":
            tickers = _fetch_sp500()
    except Exception as e:
        print(f"Universe fetch failed ({name}): {e}. Using fallback.")

    if not tickers:
        tickers = FALLBACK_TICKERS

    # Always include SPY benchmark
    if "SPY" not in tickers:
        tickers.append("SPY")

    pd.DataFrame({"ticker": tickers}).to_csv(cache_file, index=False)
    return tickers
