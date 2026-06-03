"""Four-stage momentum + pullback strategy logic (EMA-driven).

Buckets (in order of "actionability today"):
  Stage 1 — EMERGING: EMA20 turning up from flat + EMA5↗EMA20 cross + volume confirm.
                      Early-stage candidate, watch for trend confirmation.
  Stage 2 — STRONG:   EMA20 sloping up at healthy angle (3-7% / 10d), price above
                      EMA20 & EMA50, outperforming SPY. Confirmed uptrend.
  Stage 3 — PULLBACK: A strong stock now pulled back near EMA20 on light volume.
                      "Golden zone" = high slope AND tight distance to EMA20.
  Stage 4 — ENTRY:    Pullback held → green candle reclaims EMA20, volume returns,
                      RSI 40-60. The decisive signal.

Each stage scores 0-100. A stock is bucketed by the most-actionable stage it
qualifies for (entry > pullback > strong > emerging).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .indicators import enrich, relative_strength
from .minervini import MinerviniResult, check_trend_template, compute_rs_ratings
from .vcp import VCPResult, analyze_vcp


@dataclass
class Signal:
    ticker: str
    bucket: str           # 'entry' | 'pullback' | 'strong' | 'emerging'
    confidence: float
    price: float
    entry: float
    stop: float
    target: float
    target2: float
    r_multiple: float
    stage_scores: Dict[str, float]
    notes: str = ""
    ema20_slope: float = 0.0
    hv_pct: float = 0.0
    # Minervini Trend Template (8 conditions). Set when scan_universe runs with
    # rs_ratings; remains None if we couldn't compute it.
    minervini: Optional[MinerviniResult] = None
    vcp: Optional[VCPResult] = None

    def as_row(self) -> dict:
        row = {
            "Ticker": self.ticker,
            "Bucket": self.bucket,
            "Confidence": round(self.confidence, 1),
            "Minervini": self.minervini.badge if self.minervini else "—",
            "RS": round(self.minervini.rs_rating, 0) if self.minervini and self.minervini.rs_rating is not None else None,
            "VCP": self.vcp.badge if self.vcp else "—",
            "VCP tight%": round(self.vcp.tightness_pct, 2) if self.vcp and self.vcp.tightness_pct is not None else None,
            "距Pivot%": round(self.vcp.distance_to_pivot_pct, 2) if self.vcp and self.vcp.distance_to_pivot_pct is not None else None,
            "Pivot $": round(self.vcp.pivot_price, 2) if self.vcp and self.vcp.pivot_price is not None else None,
            "Price": round(self.price, 2),
            "Entry": round(self.entry, 2),
            "Stop": round(self.stop, 2),
            "T1": round(self.target, 2),
            "T2": round(self.target2, 2),
            "R:R (T1)": round(self.r_multiple, 2),
            "Stop %": round((self.entry - self.stop) / self.entry * 100, 2),
            "EMA20 slope %": round(self.ema20_slope, 2),
            "HV pct": round(self.hv_pct, 0) if pd.notna(self.hv_pct) else None,
            "Notes": self.notes,
        }
        return row


# ---------------------------------------------------------------------------
# Helper: EMA20 slope scoring — bell-shaped around 3-7% / 10d sweet spot
# ---------------------------------------------------------------------------
def _slope_score(slope_pct_10d: float, max_pts: float = 30) -> float:
    """Score the EMA20 10-day slope %. Bell curve peaks at 4-6%.

      0%   → 0  (flat, no trend)
      2%   → 60% of max (mild)
      3-7% → 100% of max (the 45° sweet spot)
      8-12%→ 70% (steep, parabolic risk)
      >15% → 30% (too parabolic, likely revert)
      <0%  → 0 (downtrend, disqualify)
    """
    if pd.isna(slope_pct_10d) or slope_pct_10d <= 0:
        return 0.0
    s = float(slope_pct_10d)
    if s < 2:
        return max_pts * (s / 2) * 0.6
    if s <= 7:
        # Linear ramp from 60% (at 2%) to 100% (at 4-7%)
        if s <= 4:
            return max_pts * (0.6 + 0.4 * (s - 2) / 2)
        return max_pts  # 4-7%: full marks
    if s <= 12:
        return max_pts * (1.0 - 0.3 * (s - 7) / 5)  # 100% → 70%
    if s <= 20:
        return max_pts * (0.7 - 0.4 * (s - 12) / 8)  # 70% → 30%
    return max_pts * 0.3


# ---------------------------------------------------------------------------
# EMERGING — early-stage trend candidate
# ---------------------------------------------------------------------------
def score_emerging(df: pd.DataFrame, rs_vs_spy: float) -> float:
    """0-100. EMA20 just turning up + EMA5↗EMA20 cross + volume confirm.

    Looking for stocks that will BECOME strong in the next 2-4 weeks.
    Hard floor: 10-day slope must be > -2% (no strong downtrends allowed in).
    """
    if len(df) < 30:
        return 0.0
    last = df.iloc[-1]

    # Hard guard: 10-day slope must be ≥ 0 — "emerging" means the trend is
    # actually turning up, not still rolling over. A slight 5d uptick inside
    # a 10d downtrend is just noise.
    slope_10d = last.get("EMA20_slope10")
    if pd.isna(slope_10d) or slope_10d < 0:
        return 0.0

    score = 0.0

    # 30 pts — EMA20 just turning up: 5d slope clearly positive, prior was flat/weak
    if pd.notna(last["EMA20"]) and len(df) >= 25:
        slope_5d = (last["EMA20"] - df["EMA20"].iloc[-6]) / df["EMA20"].iloc[-6] * 100
        slope_20d = (df["EMA20"].iloc[-11] - df["EMA20"].iloc[-21]) / df["EMA20"].iloc[-21] * 100
        if pd.notna(slope_5d) and pd.notna(slope_20d):
            # Stricter: 5d slope > 1%, prior 20d slope was flat-to-down (≤ 1%)
            if slope_5d > 1.0 and slope_20d <= 1.0:
                turn_strength = min(1.0, slope_5d / 3.0)  # full at 3%/5d
                score += 30 * turn_strength

    # 25 pts — recent EMA5↗EMA20 cross (within last 10 days)
    days_since = last.get("DaysSinceEMA5xUP", np.nan)
    if pd.notna(days_since) and days_since <= 10:
        score += 12.5 + 12.5 * (1 - days_since / 10)  # 25 if 0d, 12.5 if 10d

    # 20 pts — price above EMA20 (confirmed reclaim)
    if pd.notna(last["EMA20"]) and last["Close"] > last["EMA20"]:
        dist = (last["Close"] - last["EMA20"]) / last["EMA20"] * 100
        if dist <= 8:  # not too far above (avoid late chase)
            score += 20
        elif dist <= 15:
            score += 10

    # 15 pts — recent volume above 20d average
    recent_vol_ratio = df["VolRatio"].iloc[-5:].mean()
    if pd.notna(recent_vol_ratio) and recent_vol_ratio >= 1.0:
        score += 15 * min(1.0, (recent_vol_ratio - 0.9) / 0.5)

    # 10 pts — RS vs SPY at least positive (any momentum vs market)
    if pd.notna(rs_vs_spy) and rs_vs_spy > 0:
        score += 10 * min(1.0, rs_vs_spy / 10)

    return min(100.0, score)


# ---------------------------------------------------------------------------
# STRONG — confirmed uptrend. EMA20 slope is the centerpiece.
# ---------------------------------------------------------------------------
def score_strong(df: pd.DataFrame, rs_vs_spy: float) -> float:
    last = df.iloc[-1]
    score = 0.0

    # 30 pts — EMA20 slope (THE key indicator)
    score += _slope_score(last.get("EMA20_slope10"), max_pts=30)

    # 25 pts — Price above EMA20 AND EMA50 (both trend lines respected)
    above_ema20 = pd.notna(last["EMA20"]) and last["Close"] > last["EMA20"]
    above_ema50 = pd.notna(last["EMA50"]) and last["Close"] > last["EMA50"]
    if above_ema20 and above_ema50:
        # bonus for clean stack: EMA20 > EMA50 (proper bullish alignment)
        if last["EMA20"] > last["EMA50"]:
            score += 25
        else:
            score += 15  # above both but EMAs crossed (early stage)
    elif above_ema20:
        score += 12

    # 25 pts — Relative strength vs SPY (3M)
    if pd.notna(rs_vs_spy) and rs_vs_spy >= 0:
        score += 25 * min(1.0, rs_vs_spy / 20)

    # 10 pts — EMA50 also rising (long-term trend confirmation)
    if pd.notna(last.get("EMA50_slope10")) and last["EMA50_slope10"] > 0:
        score += 10 * min(1.0, last["EMA50_slope10"] / 3)

    # 10 pts — recent breakout-day with vol > 1.5x avg + green close
    recent = df.iloc[-10:]
    if ((recent["VolRatio"] >= 1.5) & (recent["Close"] > recent["Open"])).any():
        score += 10

    return min(100.0, score)


# ---------------------------------------------------------------------------
# PULLBACK — strong stock now sitting at EMA20 on light volume
# ---------------------------------------------------------------------------
def score_pullback(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    if pd.isna(last["EMA20"]):
        return 0.0
    score = 0.0

    # 35 pts — proximity to EMA20 (≤ 3% above is the pullback zone)
    dist = last.get("DistEMA20", np.nan)
    if pd.notna(dist):
        abs_dist = abs(dist)
        if abs_dist <= 3:
            score += 35 * (1 - abs_dist / 3)
        elif abs_dist <= 6:
            score += 18 * (1 - (abs_dist - 3) / 3)

    # 20 pts — Was extended above EMA20 in last 15 bars (proves it's a real pullback)
    look = df.iloc[-15:-1]
    if len(look) > 0:
        max_ext = ((look["Close"] - look["EMA20"]) / look["EMA20"] * 100).max()
        if pd.notna(max_ext) and max_ext > 5:
            score += 20 * min(1.0, max_ext / 15)

    # 15 pts — Pullback on declining volume (clean, not distribution)
    pullback_vol = df["VolRatio"].iloc[-5:].mean()
    if pd.notna(pullback_vol) and pullback_vol < 1.0:
        score += 15 * min(1.0, (1.0 - pullback_vol) / 0.4)

    # 15 pts — Still above EMA50 (trend intact)
    if pd.notna(last["EMA50"]) and last["Close"] > last["EMA50"]:
        score += 15

    # 15 pts — GOLDEN ZONE bonus: steep slope + tight to EMA20
    # This is the single highest-conviction pullback setup
    slope = last.get("EMA20_slope10", np.nan)
    if pd.notna(slope) and pd.notna(dist):
        if slope >= 3 and abs(dist) <= 2:
            score += 15
        elif slope >= 2 and abs(dist) <= 3:
            score += 8

    return min(100.0, score)


# ---------------------------------------------------------------------------
# ENTRY — confirmation candle: green, volume returns, RSI 40-60
# ---------------------------------------------------------------------------
def score_entry(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    conditions_met = 0

    # Condition 1 (30) — Green candle closing above EMA20 (or EMA10)
    is_green = last["Close"] > last["Open"]
    above_ema = (
        (pd.notna(last["EMA20"]) and last["Close"] > last["EMA20"]) or
        (pd.notna(last["EMA10"]) and last["Close"] > last["EMA10"])
    )
    above_prev_high = last["Close"] > prev["High"]
    if is_green and above_ema:
        score += 30
        conditions_met += 1
        if above_prev_high:
            score += 5  # bonus: engulfing-style follow-through

    # Condition 2 (30) — Volume picking up vs recent pullback days
    if pd.notna(last["VolRatio"]):
        prev_vol = df["VolRatio"].iloc[-6:-1].mean()
        if last["VolRatio"] >= 1.0 and last["VolRatio"] > prev_vol:
            score += 30 * min(1.0, last["VolRatio"] / 1.5)
            conditions_met += 1

    # Condition 3 (30) — RSI in 40-60 sweet spot
    rsi_v = last["RSI14"]
    if pd.notna(rsi_v):
        if 40 <= rsi_v <= 60:
            score += 30
            conditions_met += 1
        elif 35 <= rsi_v < 40 or 60 < rsi_v <= 65:
            score += 18
        elif 30 <= rsi_v < 35 or 65 < rsi_v <= 70:
            score += 8

    # Need at least 2 of 3 conditions hit
    if conditions_met < 2:
        score = min(score, 40)

    return min(100.0, score)


# ---------------------------------------------------------------------------
# Price targets
# ---------------------------------------------------------------------------
def compute_levels(df: pd.DataFrame, bucket: str) -> tuple[float, float, float, float]:
    """Return (entry, stop, target1, target2)."""
    last = df.iloc[-1]
    price = float(last["Close"])

    if bucket == "entry":
        entry = price
    elif bucket == "emerging":
        # Emerging: use current price (already turning up)
        entry = price
    else:
        # Strong / Pullback: planned entry on breakout above 20-day high
        entry = float(df["High"].iloc[-20:].max())

    # Swing low over last 10 days, with small buffer below
    swing_low = float(df["Low"].iloc[-10:].min())
    ideal_stop = swing_low * 0.98
    # Clamp stop into [entry*0.93, entry*0.97]: cap downside at 7%, keep room
    stop = max(entry * 0.93, min(entry * 0.97, ideal_stop))

    risk = entry - stop
    target1 = entry + 2 * risk

    # Measured move: height of last 50-day range projected upward
    range_height = float(df["High50"].iloc[-1] - df["Low20"].iloc[-1])
    if not np.isfinite(range_height) or range_height <= 0:
        range_height = entry * 0.15
    target2 = entry + range_height

    return entry, stop, target1, target2


# ---------------------------------------------------------------------------
# Per-ticker evaluation
# ---------------------------------------------------------------------------
def evaluate(
    ticker: str,
    df: pd.DataFrame,
    spy: pd.DataFrame,
) -> Optional[Signal]:
    if df is None or len(df) < 60:
        return None

    enriched = enrich(df)
    if pd.isna(enriched["EMA20"].iloc[-1]):
        return None

    spy_enriched = enrich(spy)
    rs_series = relative_strength(enriched["Close"], spy_enriched["Close"], n=63)
    rs_vs_spy = float(rs_series.iloc[-1]) if len(rs_series) else 0.0

    s_strong = score_strong(enriched, rs_vs_spy)
    s_pullback = score_pullback(enriched)
    s_entry = score_entry(enriched)
    s_emerging = score_emerging(enriched, rs_vs_spy)

    bucket: Optional[str] = None
    confidence = 0.0
    notes_parts: list[str] = []

    # Bucket precedence: entry > pullback > strong > emerging
    if s_strong >= 50 and s_pullback >= 50 and s_entry >= 60:
        bucket = "entry"
        confidence = 0.5 * s_entry + 0.25 * s_pullback + 0.25 * s_strong
        notes_parts.append("入場訊號：3階段確認")
    elif s_strong >= 50 and s_pullback >= 55:
        bucket = "pullback"
        confidence = 0.6 * s_pullback + 0.4 * s_strong
        notes_parts.append("回調中 — 等入場陽燭")
    elif s_strong >= 65:
        bucket = "strong"
        confidence = s_strong
        notes_parts.append("趨勢強勢 — 等回調")
    elif s_emerging >= 60 and s_strong < 65:
        # Emerging only if not yet strong (graduated stocks should be in 'strong')
        bucket = "emerging"
        confidence = s_emerging
        notes_parts.append("早期轉強 — 觀察是否確認趨勢")
    else:
        return None

    entry, stop, t1, t2 = compute_levels(enriched, bucket)
    if entry <= stop:
        return None

    last = enriched.iloc[-1]
    ema20_slope = float(last.get("EMA20_slope10", 0.0) or 0.0)
    hv_pct = last.get("HV_pct", np.nan)

    notes_parts.append(f"EMA20斜率 {ema20_slope:+.1f}%/10d")
    notes_parts.append(f"3M跑贏SPY {rs_vs_spy:+.1f}%")

    return Signal(
        ticker=ticker,
        bucket=bucket,
        confidence=confidence,
        price=float(last["Close"]),
        entry=entry,
        stop=stop,
        target=t1,
        target2=t2,
        r_multiple=(t1 - entry) / (entry - stop),
        stage_scores={
            "emerging": round(s_emerging, 1),
            "strong": round(s_strong, 1),
            "pullback": round(s_pullback, 1),
            "entry": round(s_entry, 1),
        },
        notes=" | ".join(notes_parts),
        ema20_slope=ema20_slope,
        hv_pct=float(hv_pct) if pd.notna(hv_pct) else float("nan"),
    )


def scan_universe(
    price_data: Dict[str, pd.DataFrame],
    benchmark_ticker: str = "SPY",
) -> list[Signal]:
    spy = price_data.get(benchmark_ticker)
    if spy is None or spy.empty:
        raise RuntimeError(f"Benchmark {benchmark_ticker} not found in price data")

    # Compute RS Ratings for the whole universe FIRST (need cross-sectional rank)
    rs_ratings = compute_rs_ratings(price_data)

    signals: list[Signal] = []
    for t, df in price_data.items():
        if t == benchmark_ticker:
            continue
        try:
            sig = evaluate(t, df, spy)
            if sig is not None:
                # Attach Minervini Trend Template check + VCP analysis
                try:
                    enriched = enrich(df)
                    sig.minervini = check_trend_template(
                        enriched, rs_rating=rs_ratings.get(t)
                    )
                except Exception:
                    pass
                try:
                    sig.vcp = analyze_vcp(df, threshold_pct=5.0, lookback_bars=120)
                except Exception:
                    pass
                signals.append(sig)
        except Exception as e:
            print(f"Evaluate failed for {t}: {e}")

    # Sort by actionability: entry → pullback → strong → emerging, then confidence
    bucket_order = {"entry": 0, "pullback": 1, "strong": 2, "emerging": 3}
    signals.sort(key=lambda s: (bucket_order.get(s.bucket, 9), -s.confidence))
    return signals
