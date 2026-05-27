"""Simple bar-by-bar backtester for the 3-stage strategy.

Walks forward through history. At each bar, evaluates the strategy on data
known *up to that bar* (no look-ahead). When an entry signal fires AND no
position is open, opens a long position. Manages the position with the
exit rules until stopped out or trend-failure exit fires.

Exit rules (all on close):
  1. Initial stop hit → exit
  2. Close below SMA20 → exit half (only once per trade)
  3. Close below SMA50 → exit full
  4. Trailing stop = max(SMA20, recent swing low) once price is up >5% from entry
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .indicators import enrich
from .strategy import evaluate


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    half_exit_date: pd.Timestamp | None = None
    half_exit_price: float | None = None
    initial_stop: float = 0.0
    confidence_at_entry: float = 0.0
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        """Blended return assuming 50% out at half_exit, 50% at exit."""
        if self.half_exit_price is not None:
            return (
                0.5 * (self.half_exit_price / self.entry_price - 1)
                + 0.5 * (self.exit_price / self.entry_price - 1)
            ) * 100
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    def as_row(self) -> dict:
        return {
            "Entry date": self.entry_date.date(),
            "Entry $": round(self.entry_price, 2),
            "Exit date": self.exit_date.date(),
            "Exit $": round(self.exit_price, 2),
            "Half exit $": round(self.half_exit_price, 2) if self.half_exit_price else None,
            "Return %": round(self.return_pct, 2),
            "Days held": self.holding_days,
            "Conf at entry": round(self.confidence_at_entry, 1),
            "Exit reason": self.exit_reason,
        }


def backtest(
    df_full: pd.DataFrame,
    spy_full: pd.DataFrame,
    ticker: str = "TICKER",
    warmup_bars: int = 80,
) -> List[Trade]:
    """Walk-forward backtest. df_full is the full OHLCV history for the ticker."""
    if len(df_full) < warmup_bars + 5:
        return []

    trades: List[Trade] = []
    in_pos = False
    entry_px = 0.0
    entry_dt = None
    stop = 0.0
    half_exited = False
    half_px = None
    half_dt = None
    conf_at_entry = 0.0

    # Pre-enrich for trailing-stop lookups (no look-ahead because we'll only
    # read .iloc[:i+1] each loop)
    enriched_full = enrich(df_full)

    for i in range(warmup_bars, len(df_full)):
        bar_slice = df_full.iloc[: i + 1]
        spy_slice = spy_full.loc[: bar_slice.index[-1]]
        if len(spy_slice) < warmup_bars:
            continue

        bar = enriched_full.iloc[i]
        date = df_full.index[i]
        close = float(bar["Close"])
        low = float(bar["Low"])

        if in_pos:
            # 1. Initial stop hit (intraday low touches stop)
            if low <= stop:
                # Assume filled at stop
                trades.append(Trade(
                    entry_date=entry_dt, entry_price=entry_px,
                    exit_date=date, exit_price=stop,
                    half_exit_date=half_dt, half_exit_price=half_px,
                    initial_stop=stop, confidence_at_entry=conf_at_entry,
                    exit_reason="Stop hit",
                ))
                in_pos = False
                continue

            sma20 = bar["SMA20"]
            sma50 = bar["SMA50"]

            # 3. Full exit: close below SMA50
            if pd.notna(sma50) and close < sma50:
                trades.append(Trade(
                    entry_date=entry_dt, entry_price=entry_px,
                    exit_date=date, exit_price=close,
                    half_exit_date=half_dt, half_exit_price=half_px,
                    initial_stop=stop, confidence_at_entry=conf_at_entry,
                    exit_reason="Close < SMA50",
                ))
                in_pos = False
                continue

            # 2. Half exit on first close < SMA20
            if not half_exited and pd.notna(sma20) and close < sma20:
                half_exited = True
                half_px = close
                half_dt = date

            # 4. Trail stop up once profit > 5%: max(SMA20, 10-day swing low * 0.98)
            if close / entry_px - 1 > 0.05:
                swing_low_10 = float(df_full["Low"].iloc[max(0, i - 10): i + 1].min())
                trail = max(stop, swing_low_10 * 0.98)
                if pd.notna(sma20):
                    trail = max(trail, float(sma20) * 0.99)
                stop = trail

        else:
            # Look for entry signal
            try:
                sig = evaluate(ticker, bar_slice, spy_slice)
            except Exception:
                sig = None
            if sig is not None and sig.bucket == "entry":
                in_pos = True
                entry_px = close
                entry_dt = date
                stop = sig.stop
                half_exited = False
                half_px = None
                half_dt = None
                conf_at_entry = sig.confidence

    # Close any open position at the last bar
    if in_pos:
        last_bar = df_full.iloc[-1]
        trades.append(Trade(
            entry_date=entry_dt, entry_price=entry_px,
            exit_date=df_full.index[-1], exit_price=float(last_bar["Close"]),
            half_exit_date=half_dt, half_exit_price=half_px,
            initial_stop=stop, confidence_at_entry=conf_at_entry,
            exit_reason="End of data (still open)",
        ))

    return trades


def summarize(trades: List[Trade]) -> dict:
    if not trades:
        return {
            "trades": 0, "win_rate": 0.0, "avg_return": 0.0,
            "best": 0.0, "worst": 0.0, "total_return": 0.0,
            "avg_days": 0.0,
        }
    returns = np.array([t.return_pct for t in trades])
    wins = returns > 0
    # Compound return assuming all-in / all-out, equal capital each trade
    compound = float(np.prod(1 + returns / 100) - 1) * 100
    return {
        "trades": len(trades),
        "win_rate": float(wins.mean() * 100),
        "avg_return": float(returns.mean()),
        "best": float(returns.max()),
        "worst": float(returns.min()),
        "total_return": compound,
        "avg_days": float(np.mean([t.holding_days for t in trades])),
    }
