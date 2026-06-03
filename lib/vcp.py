"""Volatility Contraction Pattern (VCP) detection — Minervini's signature setup.

A textbook VCP is a series of progressively shallower pullbacks on declining
volume, terminating in a tight base with a clear pivot (resistance) above.

Detection pipeline:
  1. Zigzag pivots — find local highs/lows where price moves ≥ threshold% from
     the last pivot. Threshold defaults to 5% but is configurable per stock
     ($1000 stock vs $20 stock have very different normal volatility).
  2. Contractions — pair each High with the immediately following Low.
     A valid VCP requires each contraction to be ≤ 0.6 × prior contraction
     (i.e. each leg ≥40% shallower than the previous).
  3. Volume decline — avg volume in each contraction window should fall.
     Final contraction's volume should be below 50-day MA (sellers exhausted).
  4. Pivot — the high of the most recent contraction. Buy when price breaks
     above this on volume.
  5. Tightness — average daily true range over last 10-15 days / price,
     in percent. Target < 3% means the base is properly coiled.

Maturity rating:
  🟢 Ready    — ≥3 contractions, volumes decreasing, tightness <3%, <3% from pivot
  🟡 Forming  — ≥2 contractions with volumes decreasing
  ⚪ Not yet  — doesn't meet either bar
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Contraction:
    """One H → L leg in the VCP series."""
    high_idx: int
    high_date: pd.Timestamp
    high_price: float
    low_idx: int
    low_date: pd.Timestamp
    low_price: float
    depth_pct: float
    avg_volume: float
    days: int

    def as_row(self) -> dict:
        return {
            "起點": self.high_date.date().isoformat(),
            "終點": self.low_date.date().isoformat(),
            "高點 $": round(self.high_price, 2),
            "低點 $": round(self.low_price, 2),
            "深度 %": round(self.depth_pct, 2),
            "天數": self.days,
            "平均量": int(self.avg_volume),
        }


@dataclass
class VCPResult:
    contractions: List[Contraction] = field(default_factory=list)   # Trimmed VCP series only
    all_contractions: List[Contraction] = field(default_factory=list)  # All zigzag-derived contractions
    pivot_price: Optional[float] = None
    distance_to_pivot_pct: Optional[float] = None  # negative = below pivot
    tightness_pct: Optional[float] = None
    volume_decreasing: bool = False
    contraction_decreasing: bool = False    # each ≤ 0.6 × prior (auto-true once trimmed)
    final_vol_below_50d: bool = False
    threshold_used: float = 5.0
    last_close: float = 0.0

    @property
    def n_contractions(self) -> int:
        return len(self.contractions)

    @property
    def maturity(self) -> str:
        """🟢 Ready · 🟡 Forming · 🚀 Broken out · ⚪ Not yet.

        Price must be in/near the base (not already exploded past pivot) for
        Ready/Forming. If price > pivot + 2%, the VCP has broken out — useful
        signal but no longer a setup to plan around.
        """
        d = self.distance_to_pivot_pct

        # Already broken above — flag separately
        if (
            d is not None and d > 2.0
            and self.n_contractions >= 2
            and self.volume_decreasing
            and d <= 15.0  # only flag fresh breakouts, not stale ones
        ):
            return "🚀 Broken out"

        # Must be at/near pivot to be a setup
        if d is None or d > 2.0 or d < -15.0:
            return "⚪ Not yet"

        if (
            self.n_contractions >= 3
            and self.volume_decreasing
            and self.tightness_pct is not None and self.tightness_pct < 3.0
            and -3.0 <= d <= 2.0
        ):
            return "🟢 Ready"
        if (
            self.n_contractions >= 2
            and self.volume_decreasing
            and -10.0 <= d <= 2.0
        ):
            return "🟡 Forming"
        return "⚪ Not yet"

    @property
    def badge(self) -> str:
        m = self.maturity
        if "Ready" in m:
            return f"🟢 Ready ({self.n_contractions}T, tight {self.tightness_pct:.1f}%)"
        if "Forming" in m:
            return f"🟡 {self.n_contractions}T 收緊中"
        if "Broken" in m:
            return f"🚀 已突破 +{self.distance_to_pivot_pct:.1f}%"
        return "⚪ 無VCP"

    def as_summary_row(self) -> dict:
        return {
            "VCP": self.badge,
            "VCP 收窄度 %": round(self.tightness_pct, 2) if self.tightness_pct is not None else None,
            "距 Pivot %": round(self.distance_to_pivot_pct, 2) if self.distance_to_pivot_pct is not None else None,
            "Pivot $": round(self.pivot_price, 2) if self.pivot_price is not None else None,
        }


# ---------------------------------------------------------------------------
def find_zigzag_pivots(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
) -> List[tuple[int, pd.Timestamp, float, str]]:
    """Find alternating high/low pivots where each move ≥ threshold% reversal.

    Uses High for new tops and Low for new bottoms — captures the true swing
    extremes rather than just Close-to-Close.

    Returns list of (index_position, date, price, 'high'|'low') in chronological order.
    The final unconfirmed pivot (still extending) is NOT included.
    """
    if df is None or len(df) < 3:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    pivots: List[tuple[int, pd.Timestamp, float, str]] = []

    # Initialize: track running extremes; direction emerges from first move
    extreme_high = highs[0]
    extreme_high_idx = 0
    extreme_low = lows[0]
    extreme_low_idx = 0
    direction: Optional[str] = None  # 'up' = looking for tops, 'down' = looking for bottoms

    for i in range(1, len(df)):
        h = highs[i]
        l = lows[i]

        if direction == "up" or direction is None:
            # Track new high
            if h > extreme_high:
                extreme_high = h
                extreme_high_idx = i
                direction = "up"
            # Check for reversal: did we drop ≥ threshold from the running high?
            drop_pct = (extreme_high - l) / extreme_high * 100
            if drop_pct >= threshold_pct and extreme_high_idx < i:
                # Confirm the top
                pivots.append((extreme_high_idx, dates[extreme_high_idx],
                               float(extreme_high), "high"))
                # Switch to looking for bottom; reset low tracking
                extreme_low = l
                extreme_low_idx = i
                direction = "down"
                continue

        if direction == "down":
            if l < extreme_low:
                extreme_low = l
                extreme_low_idx = i
            rise_pct = (h - extreme_low) / extreme_low * 100
            if rise_pct >= threshold_pct and extreme_low_idx < i:
                pivots.append((extreme_low_idx, dates[extreme_low_idx],
                               float(extreme_low), "low"))
                extreme_high = h
                extreme_high_idx = i
                direction = "up"
                continue

    return pivots


def _build_contractions(
    df: pd.DataFrame,
    pivots: List[tuple[int, pd.Timestamp, float, str]],
) -> List[Contraction]:
    """Pair each High with the immediately-following Low. Skip orphan pivots."""
    contractions: List[Contraction] = []
    volume = df["Volume"].values
    i = 0
    while i < len(pivots) - 1:
        idx_h, date_h, price_h, type_h = pivots[i]
        if type_h != "high":
            i += 1
            continue
        # Find the next low after this high
        nxt = pivots[i + 1]
        idx_l, date_l, price_l, type_l = nxt
        if type_l != "low":
            i += 1
            continue
        depth = (price_h - price_l) / price_h * 100
        avg_vol = float(np.nanmean(volume[idx_h : idx_l + 1])) if idx_l >= idx_h else 0.0
        contractions.append(Contraction(
            high_idx=idx_h, high_date=date_h, high_price=price_h,
            low_idx=idx_l, low_date=date_l, low_price=price_l,
            depth_pct=depth, avg_volume=avg_vol,
            days=int(idx_l - idx_h),
        ))
        i += 2
    return contractions


def _compute_tightness(df: pd.DataFrame, n: int = 12) -> float:
    """Average true daily range over last n bars, as % of close."""
    if len(df) < n:
        return float("nan")
    tail = df.iloc[-n:]
    rng = (tail["High"] - tail["Low"]) / tail["Close"]
    return float(rng.mean() * 100)


def _check_decreasing_volumes(contractions: List[Contraction]) -> bool:
    """True if avg volume strictly decreases across contractions."""
    if len(contractions) < 2:
        return False
    return all(
        contractions[i].avg_volume < contractions[i - 1].avg_volume
        for i in range(1, len(contractions))
    )


def _check_decreasing_contractions(contractions: List[Contraction]) -> bool:
    """True if each contraction depth is ≤ 0.6 × prior (40%+ shallower)."""
    if len(contractions) < 2:
        return False
    return all(
        contractions[i].depth_pct <= contractions[i - 1].depth_pct * 0.6
        for i in range(1, len(contractions))
    )


def _trim_to_vcp_series(all_contractions: List[Contraction]) -> List[Contraction]:
    """From all zigzag-derived contractions, keep only the most-recent tail
    that forms a valid VCP series (each step ≤ 0.6 × prior).

    Walk backwards from the latest contraction. Include the prior one only if
    it's deep enough to be a valid T_{n-1} for the next-in-series. Stop at
    the first qualification break.
    """
    if not all_contractions:
        return []
    series = [all_contractions[-1]]
    for i in range(len(all_contractions) - 2, -1, -1):
        prior = all_contractions[i]
        next_in_series = series[0]
        # Forward VCP rule: next ≤ prior × 0.6  ⟺  prior ≥ next / 0.6
        if prior.depth_pct >= next_in_series.depth_pct / 0.6:
            series.insert(0, prior)
        else:
            break
    return series


def analyze_vcp(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
    lookback_bars: int = 120,
    tightness_window: int = 12,
) -> VCPResult:
    """Run full VCP analysis on the trailing window of df.

    Args:
        df: OHLCV frame with DatetimeIndex.
        threshold_pct: zigzag reversal threshold. Smaller catches shallower
            T3/T4 but more noise. 5% is a reasonable starting default for $50-500
            stocks; try 3% for low-volatility names and 7-8% for high-vol speculation.
        lookback_bars: number of recent bars to analyze (~6 months default).
        tightness_window: bars used for tightness score (10-15 typical).
    """
    res = VCPResult(threshold_used=threshold_pct)

    if df is None or df.empty:
        return res

    window = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df.copy()
    if len(window) < 30:
        return res

    res.last_close = float(window["Close"].iloc[-1])

    pivots = find_zigzag_pivots(window, threshold_pct=threshold_pct)
    all_contractions = _build_contractions(window, pivots)
    res.all_contractions = all_contractions

    # Extract only the most-recent VCP series (where each step ≤ 0.6 × prior)
    vcp_series = _trim_to_vcp_series(all_contractions)
    res.contractions = vcp_series

    res.tightness_pct = _compute_tightness(window, n=tightness_window)
    # contraction_decreasing is automatically True for a trimmed series of ≥2
    res.contraction_decreasing = len(vcp_series) >= 2
    res.volume_decreasing = _check_decreasing_volumes(vcp_series)

    # Pivot = high of the MOST RECENT contraction in the VCP series
    if vcp_series:
        last = vcp_series[-1]
        res.pivot_price = last.high_price
        res.distance_to_pivot_pct = (res.last_close - last.high_price) / last.high_price * 100

        # Final vol below 50-day MA (sellers exhausted)?
        if len(df) >= 50 and last.avg_volume > 0:
            vol50 = float(df["Volume"].iloc[-50:].mean())
            res.final_vol_below_50d = last.avg_volume < vol50

    return res
