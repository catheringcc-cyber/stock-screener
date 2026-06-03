"""Technical indicators — vectorized pandas implementations.

Primary trend line is EMA20 (reacts faster than SMA for swing trades).
SMA50 kept as a slower, structural reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with span=n (alpha = 2/(n+1))."""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def relative_strength(price: pd.Series, benchmark: pd.Series, n: int = 63) -> pd.Series:
    """% outperformance vs benchmark over last n trading days (~3 months)."""
    aligned = pd.concat([price, benchmark], axis=1).dropna()
    if len(aligned) < n + 1:
        return pd.Series(dtype=float)
    p_ret = aligned.iloc[:, 0].pct_change(n)
    b_ret = aligned.iloc[:, 1].pct_change(n)
    return (p_ret - b_ret) * 100


def slope_pct(series: pd.Series, lookback: int = 10) -> pd.Series:
    """% change of the series over `lookback` bars (a proxy for slope angle).

    Used on EMA20 — this is the single highest-alpha indicator for swing trade
    entry timing. Sweet spot is ~3-7% per 10 days (a steady, walkable uptrend).
    """
    prev = series.shift(lookback)
    return (series - prev) / prev * 100


def hv_percentile(close: pd.Series, window: int = 20, lookback: int = 252) -> pd.Series:
    """Realized-volatility percentile (0-100) of `window`-day stdev within the
    trailing `lookback` bars. Stand-in for IV rank until options data is added.

    Low percentile (<30) → quiet stock, options usually cheap.
    High percentile (>70) → noisy/expensive options.
    """
    rets = close.pct_change()
    hv = rets.rolling(window).std() * np.sqrt(252) * 100  # annualized vol %
    # percentile rank within trailing lookback
    return hv.rolling(lookback).apply(
        lambda x: (x[-1] >= x).mean() * 100 if len(x) == lookback else np.nan,
        raw=True,
    )


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns. EMA-first, SMA kept for structural reference."""
    out = df.copy()

    # Primary EMAs — EMA20 is the trend line, EMA5 the fast trigger
    out["EMA5"] = ema(out["Close"], 5)
    out["EMA10"] = ema(out["Close"], 10)
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)

    # SMA kept as structural anchors. SMA150 + SMA200 used by Minervini
    # Trend Template; SMA50 used for the 50-day MA exit rule.
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA150"] = sma(out["Close"], 150)
    out["SMA200"] = sma(out["Close"], 200)

    # EMA20 slope — THE key swing-trade indicator (45° feel ≈ 3-7% / 10d)
    out["EMA20_slope10"] = slope_pct(out["EMA20"], 10)
    out["EMA50_slope10"] = slope_pct(out["EMA50"], 10)

    # EMA5 ↗ EMA20 crossover detection (golden cross of the fast pair)
    cross_up = (out["EMA5"] > out["EMA20"]) & (out["EMA5"].shift(1) <= out["EMA20"].shift(1))
    # Distance in days since the last upward cross (capped at 60)
    cross_idx = cross_up[cross_up].index
    if len(cross_idx):
        days_since = pd.Series(index=out.index, dtype=float)
        last_cross = None
        for d in out.index:
            if d in cross_idx:
                last_cross = d
            days_since[d] = (d - last_cross).days if last_cross is not None else np.nan
        out["DaysSinceEMA5xUP"] = days_since.clip(upper=60)
    else:
        out["DaysSinceEMA5xUP"] = np.nan

    # Distance from EMA20 (% above/below) — the "are we at the pullback zone?" metric
    out["DistEMA20"] = (out["Close"] - out["EMA20"]) / out["EMA20"] * 100

    # Volatility
    out["HV_pct"] = hv_percentile(out["Close"])

    # RSI
    out["RSI14"] = rsi(out["Close"], 14)

    # Volume
    out["VolAvg20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["VolRatio"] = out["Volume"] / out["VolAvg20"]

    # Returns + range helpers
    out["Ret1M"] = out["Close"].pct_change(21) * 100
    out["Ret3M"] = out["Close"].pct_change(63) * 100
    out["High20"] = out["High"].rolling(20).max()
    out["Low20"] = out["Low"].rolling(20).min()
    out["High50"] = out["High"].rolling(50).max()

    return out
