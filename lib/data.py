"""Batched yfinance price downloader with per-day disk cache.

yfinance.download() accepts a space-separated list and returns a MultiIndex
DataFrame. We batch in groups of ~80 to balance throughput and reliability.
Same-day repeat requests are served from disk cache.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 80
LOOKBACK_DAYS = 400  # Need >200d for 200d MA + 50d MA + comfortable buffer


def _cache_path(ticker: str, asof: dt.date) -> Path:
    return CACHE_DIR / f"{ticker}_{asof.isoformat()}.parquet"


def _load_cached(ticker: str, asof: dt.date, min_start: dt.date) -> pd.DataFrame | None:
    """Return cached frame only if it covers back to min_start."""
    p = _cache_path(ticker, asof)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty:
        return None
    earliest = df.index.min().date()
    if earliest > min_start:
        # Cache is too short for this request — invalidate
        return None
    return df


def _save_cached(ticker: str, asof: dt.date, df: pd.DataFrame) -> None:
    """Merge with any existing cache to preserve the deepest history seen today."""
    p = _cache_path(ticker, asof)
    try:
        if p.exists():
            old = pd.read_parquet(p)
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(p)
    except Exception:
        pass


def download_prices(
    tickers: Iterable[str],
    lookback_days: int = LOOKBACK_DAYS,
    asof: dt.date | None = None,
) -> Dict[str, pd.DataFrame]:
    """Return dict mapping ticker -> OHLCV DataFrame indexed by date.

    Uses today's UTC date as cache key. Tickers already cached for today are
    skipped from the network call.
    """
    asof = asof or dt.datetime.utcnow().date()
    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
    min_start = asof - dt.timedelta(days=lookback_days)

    result: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []
    for t in tickers:
        cached = _load_cached(t, asof, min_start)
        if cached is not None and not cached.empty:
            result[t] = cached
        else:
            to_fetch.append(t)

    if not to_fetch:
        return result

    end = asof + dt.timedelta(days=1)
    start = min_start

    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i : i + BATCH_SIZE]
        try:
            raw = yf.download(
                tickers=" ".join(batch),
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            print(f"Batch download failed for {batch[:3]}...: {e}")
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            # Columns are (ticker, field) when group_by='ticker'
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t].dropna(how="all")
                if df.empty:
                    continue
                result[t] = df
                _save_cached(t, asof, df)
        else:
            # Single-ticker, flat columns
            t = batch[0]
            df = raw.dropna(how="all")
            if not df.empty:
                result[t] = df
                _save_cached(t, asof, df)

    return result


def download_single(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Convenience for detail page — always returns fresh long-range data."""
    data = download_prices([ticker], lookback_days=lookback_days)
    return data.get(ticker)
