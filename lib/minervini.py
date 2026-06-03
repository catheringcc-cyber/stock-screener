"""Minervini's Trend Template — 8 conditions from《Think & Trade Like a Champion》.

Minervini本人講過：「過 7 條唔夠，partial match嘅成功率大幅下降」。所以 all-8 才算
真正合格。呢個係swing trade嘅第一關gate keeper — 用嚟filter掉「趨勢有瑕疵」嘅股票。

8 個條件：
  1. 股價 > SMA150
  2. 股價 > SMA200
  3. SMA150 > SMA200
  4. SMA200 向上至少 1 個月（21個交易日）
  5. SMA50 > SMA150 同 SMA200，而且股價 > SMA50
  6. 股價 ≥ 52週低位 × 1.30（離底至少 30%）
  7. 股價 ≥ 52週高位 × 0.75（離頂在 25% 範圍內）
  8. RS Rating ≥ 70（跑贏市場 70% 以上股票）

RS Rating用IBD式weighted 12個月performance percentile rank（universe內排名）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


CONDITION_LABELS = {
    1: "股價 > 150日均線",
    2: "股價 > 200日均線",
    3: "150日線 > 200日線",
    4: "200日線向上至少1個月",
    5: "50日線 > 150日/200日線，股價 > 50日線",
    6: "股價離52週低位 ≥ 30%",
    7: "股價喺52週高位嘅25%以內",
    8: "RS Rating ≥ 70",
}


@dataclass
class MinerviniResult:
    conditions: Dict[int, bool] = field(default_factory=dict)
    passed: int = 0                       # 0-8
    rs_rating: Optional[float] = None
    price: float = 0.0
    sma50: Optional[float] = None
    sma150: Optional[float] = None
    sma200: Optional[float] = None
    sma200_21d_ago: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    pct_from_52w_low: Optional[float] = None
    pct_below_52w_high: Optional[float] = None

    @property
    def all_pass(self) -> bool:
        return self.passed == 8

    @property
    def badge(self) -> str:
        """Quick badge for screener column."""
        if self.passed == 8:
            return "🏆 8/8"
        if self.passed >= 7:
            return f"⭐ {self.passed}/8"
        if self.passed >= 5:
            return f"⚠️ {self.passed}/8"
        return f"❌ {self.passed}/8"


def compute_rs_ratings(price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """IBD-style RS Rating (1-99 percentile rank) for each ticker.

    Formula: weighted return = 0.4·r3m + 0.2·r6m + 0.2·r9m + 0.2·r12m
    Then percentile-rank within the scanned universe.

    Tickers without 252 trading days of history get None (filtered out of ranking).
    """
    raw: Dict[str, float] = {}
    for ticker, df in price_data.items():
        if df is None or len(df) < 252:
            continue
        close = df["Close"]
        last = float(close.iloc[-1])
        try:
            r3m = last / float(close.iloc[-63]) - 1
            r6m = last / float(close.iloc[-126]) - 1
            r9m = last / float(close.iloc[-189]) - 1
            r12m = last / float(close.iloc[-252]) - 1
        except (IndexError, ZeroDivisionError):
            continue
        if not all(np.isfinite([r3m, r6m, r9m, r12m])):
            continue
        raw[ticker] = 0.4 * r3m + 0.2 * r6m + 0.2 * r9m + 0.2 * r12m

    if not raw:
        return {}

    series = pd.Series(raw)
    # rank(pct=True) gives 0-1; scale to 1-99 (IBD convention)
    ratings = (series.rank(pct=True) * 98 + 1).round(0)
    return ratings.to_dict()


def check_trend_template(
    df_enriched: pd.DataFrame,
    rs_rating: Optional[float] = None,
) -> MinerviniResult:
    """Run all 8 Minervini conditions on an enriched OHLCV frame.

    df_enriched must already have SMA50/SMA150/SMA200 columns (use indicators.enrich).
    """
    res = MinerviniResult(conditions={i: False for i in range(1, 9)}, rs_rating=rs_rating)

    if df_enriched is None or df_enriched.empty:
        return res

    last = df_enriched.iloc[-1]
    res.price = float(last["Close"])

    sma50 = last.get("SMA50")
    sma150 = last.get("SMA150")
    sma200 = last.get("SMA200")
    res.sma50 = float(sma50) if pd.notna(sma50) else None
    res.sma150 = float(sma150) if pd.notna(sma150) else None
    res.sma200 = float(sma200) if pd.notna(sma200) else None

    # Condition 1: price > SMA150
    if res.sma150 is not None:
        res.conditions[1] = res.price > res.sma150

    # Condition 2: price > SMA200
    if res.sma200 is not None:
        res.conditions[2] = res.price > res.sma200

    # Condition 3: SMA150 > SMA200
    if res.sma150 is not None and res.sma200 is not None:
        res.conditions[3] = res.sma150 > res.sma200

    # Condition 4: SMA200 rising at least 21 trading days
    if len(df_enriched) >= 22 and pd.notna(sma200):
        prev = df_enriched["SMA200"].iloc[-22]
        if pd.notna(prev) and prev > 0:
            res.sma200_21d_ago = float(prev)
            res.conditions[4] = float(sma200) > float(prev)

    # Condition 5: SMA50 > SMA150 AND SMA50 > SMA200 AND price > SMA50
    if all(v is not None for v in (res.sma50, res.sma150, res.sma200)):
        res.conditions[5] = (
            res.sma50 > res.sma150
            and res.sma50 > res.sma200
            and res.price > res.sma50
        )

    # 52-week window (252 trading days)
    if len(df_enriched) >= 252:
        window = df_enriched["Close"].iloc[-252:]
        res.week52_high = float(window.max())
        res.week52_low = float(window.min())
        if res.week52_low > 0:
            res.pct_from_52w_low = (res.price / res.week52_low - 1) * 100
            res.conditions[6] = res.price >= res.week52_low * 1.30
        if res.week52_high > 0:
            res.pct_below_52w_high = (1 - res.price / res.week52_high) * 100
            res.conditions[7] = res.price >= res.week52_high * 0.75

    # Condition 8: RS Rating ≥ 70
    if rs_rating is not None and pd.notna(rs_rating):
        res.conditions[8] = rs_rating >= 70

    res.passed = sum(res.conditions.values())
    return res
