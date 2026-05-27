"""期權異動分析 — 偵測異常 options activity（insider tells）。

策略：
- 只掃 **60日內到期** 嘅expiries（大戶搏短期事件，唔買長expiry）
- 每張contract計 **Vol / Open Interest ratio** — 今日成交相對於現有未平倉
  - Vol/OI >= 2  : 異常（今日有大倉新開）
  - Vol/OI >= 5  : 高度可疑（典型insider pattern）
- 配合 PCR（Put/Call Volume Ratio）判斷整體情緒：
  - PCR > 1.5  : 看淡傾斜（hedging or insider bet on drop）
  - PCR < 0.4  : 看好傾斜
- 每日snapshot保存到 .cache/options/{ticker}_{date}.parquet
  → 經過一段時間就有歷史baseline可計z-score
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "options"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 約120隻最活躍options名（mega cap tech + speculation + financials + ETFs）
HOT_TICKERS = [
    # Mega cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "NFLX",
    "ORCL", "CRM", "ADBE", "QCOM", "INTU", "AMD", "INTC", "MU", "AMAT", "LRCX",
    # High-IV speculation / momentum
    "ARM", "PLTR", "COIN", "MSTR", "HOOD", "SOFI", "AFRM", "RBLX", "DKNG",
    "SMCI", "ASTS", "RKLB", "SHOP", "UBER", "ABNB", "RIVN", "LCID", "NIO", "FUTU",
    # Semis & AI infra
    "TSM", "ASML", "ANET", "DELL", "VRT", "MRVL", "ON", "MCHP",
    # Cybersecurity & SaaS
    "PANW", "CRWD", "SNOW", "DDOG", "NET", "MDB", "ZS", "OKTA", "WDAY", "NOW",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "PYPL",
    "SQ", "BRK-B",
    # Healthcare & biotech
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "VRTX", "REGN",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC",
    # Consumer
    "HD", "LOW", "WMT", "COST", "TGT", "NKE", "MCD", "SBUX", "DIS", "F", "GM",
    # Industrials & defense
    "BA", "CAT", "DE", "GE", "HON", "RTX", "LMT", "NOC",
    # China ADRs (often have unusual flow)
    "BABA", "JD", "PDD", "NIO", "LI", "XPEV", "BIDU",
    # Major ETFs
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLE", "XLF", "XLK",
]
# Dedup
HOT_TICKERS = list(dict.fromkeys(HOT_TICKERS))

VOL_OI_THRESHOLD = 2.0           # Vol/OI ≥ 2 = unusual
VOL_OI_HIGH = 5.0                # Vol/OI ≥ 5 = highly suspicious
MIN_CONTRACT_VOL = 200           # Min today's volume to consider for Vol/OI
MIN_CONTRACT_OI = 50             # Min existing OI for Vol/OI signal
LARGE_NOTIONAL_MIN_VOL = 500     # For "large $ flow" path
LARGE_NOTIONAL_MIN_USD = 200000  # $200K+ notional flow
MAX_EXPIRY_DAYS = 60             # Only look at expiries within 60 days

# Speculation-range filters: cut out arb/box trades + lottery tickets
# Positive otm_pct = OTM. For PUTs: strike below spot = OTM (positive value).
# For CALLs: strike above spot = OTM (positive value).
MIN_OTM_PCT = -10                # Skip ITM beyond -10% (likely arb / boxes)
MAX_OTM_PCT = 40                 # Skip too-far OTM (>40% = pure lottery)
MIN_IV = 0.05                    # 5% — below = data error
MAX_IV = 5.0                     # 500% — above = data error


@dataclass
class UnusualContract:
    ticker: str
    expiry: str
    days_to_expiry: int
    type: str               # 'call' or 'put'
    strike: float
    underlying: float
    volume: int
    open_interest: int
    vol_oi_ratio: float
    iv: float               # implied volatility (0-1)
    last_price: float
    dollar_notional: float  # volume * lastPrice * 100 (per contract = 100 shares)
    otm_pct: float          # % distance OTM (positive = OTM)
    flag: str = "vol_oi"    # 'vol_oi' (confirmed Vol/OI signal) | 'big_flow' (large $ flow, OI unreliable)

    def as_row(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Type": "PUT" if self.type == "put" else "CALL",
            "Strike": round(self.strike, 2),
            "Expiry": self.expiry,
            "DTE": self.days_to_expiry,
            "Vol": self.volume,
            "OI": self.open_interest,
            "Vol/OI": round(self.vol_oi_ratio, 1) if self.vol_oi_ratio > 0 else None,
            "IV": round(self.iv * 100, 1) if pd.notna(self.iv) else None,
            "OTM%": round(self.otm_pct, 1),
            "$ Notional": round(self.dollar_notional, 0),
            "Signal": "Vol/OI異動" if self.flag == "vol_oi" else "大額流入",
        }


@dataclass
class TickerOptionsSummary:
    ticker: str
    underlying: float
    n_contracts_scanned: int
    call_volume: int
    put_volume: int
    pcr: float                           # put/call volume ratio
    call_oi: int
    put_oi: int
    pcr_oi: float                        # PCR based on OI
    unusual: List[UnusualContract] = field(default_factory=list)
    max_vol_oi: float = 0.0
    sentiment: str = "neutral"           # 'bullish' | 'bearish' | 'neutral' (PCR-based)
    # Historical baseline + z-scores (filled in from snapshots if available)
    baseline_days: int = 0               # how many days of history were used
    pcr_baseline: float = 0.0
    pcr_zscore: float = 0.0              # (today_pcr - mean) / std
    vol_baseline: float = 0.0            # total options volume baseline
    vol_zscore: float = 0.0
    # Smart money direction (weighted by $ notional + DTE + OTM)
    smart_money: str = "mixed"           # 'bullish' | 'bearish' | 'mixed' | 'none'
    smart_money_conviction: float = 0.0  # 0-100
    smart_money_label: str = ""          # 中文 readable tag
    error: Optional[str] = None

    def anomaly_score(self) -> float:
        """0-100 score for how unusual this ticker's options activity is."""
        if self.error or self.n_contracts_scanned == 0:
            return 0.0
        score = 0.0
        # 35 pts — max Vol/OI ratio (gold standard signal)
        if self.max_vol_oi >= VOL_OI_HIGH:
            score += 35
        elif self.max_vol_oi >= VOL_OI_THRESHOLD:
            score += 35 * (self.max_vol_oi - VOL_OI_THRESHOLD) / (VOL_OI_HIGH - VOL_OI_THRESHOLD)
        # 25 pts — PCR extreme (or PCR z-score when baseline available)
        if self.baseline_days >= 3 and abs(self.pcr_zscore) >= 2:
            # z-score path: |z| ≥ 2 is statistical anomaly (more meaningful than absolute PCR)
            score += 25 * min(1.0, (abs(self.pcr_zscore) - 2) / 2 + 0.5)
        elif self.pcr > 1.5 or self.pcr < 0.4:
            extreme = max(self.pcr / 1.5, 0.4 / self.pcr) if self.pcr > 0 else 1
            score += 25 * min(1.0, (extreme - 1) / 1.5)
        # 15 pts — total options volume z-score (whole-name attention spike)
        if self.baseline_days >= 3 and self.vol_zscore >= 2:
            score += 15 * min(1.0, (self.vol_zscore - 2) / 2 + 0.5)
        # 15 pts — unusual contract count
        score += min(15, len(self.unusual) * 2)
        # 10 pts — total $ notional of unusual contracts
        total_notional = sum(u.dollar_notional for u in self.unusual)
        if total_notional >= 1_000_000:
            score += 10
        elif total_notional >= 100_000:
            score += 10 * total_notional / 1_000_000
        return min(100.0, score)


# ---------------------------------------------------------------------------
def _select_near_expiries(yf_ticker: yf.Ticker, max_days: int = MAX_EXPIRY_DAYS) -> List[str]:
    """Return only expiries within max_days from today."""
    try:
        expiries = yf_ticker.options
    except Exception:
        return []
    today = dt.date.today()
    out = []
    for e in expiries:
        try:
            ed = dt.date.fromisoformat(e)
        except ValueError:
            continue
        days = (ed - today).days
        if 0 <= days <= max_days:
            out.append(e)
    return out


def _process_chain(
    ticker: str,
    expiry: str,
    chain,                  # yf.OptionChain namedtuple (calls, puts)
    underlying: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = dt.date.today()
    ed = dt.date.fromisoformat(expiry)
    dte = (ed - today).days

    def _enrich(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        x = df.copy()
        x["type"] = opt_type
        x["expiry"] = expiry
        x["dte"] = dte
        x["underlying"] = underlying
        # Yahoo sometimes has NaN volume/OI
        x["volume"] = x["volume"].fillna(0).astype(int)
        x["openInterest"] = x["openInterest"].fillna(0).astype(int)
        # Vol/OI: divide by OI when meaningful, else 0 (not 999 — newly-listed
        # contracts with OI=0 give false signals).
        x["vol_oi"] = np.where(
            x["openInterest"] > 0,
            x["volume"] / x["openInterest"],
            0.0,
        )
        if opt_type == "call":
            x["otm_pct"] = (x["strike"] - underlying) / underlying * 100
        else:
            x["otm_pct"] = (underlying - x["strike"]) / underlying * 100
        x["dollar_notional"] = x["volume"] * x["lastPrice"] * 100
        return x

    return _enrich(chain.calls, "call"), _enrich(chain.puts, "put")


# ---------------------------------------------------------------------------
# Historical baseline from saved snapshots
# ---------------------------------------------------------------------------
def compute_baseline(ticker: str, lookback_days: int = 20) -> dict:
    """Read past snapshot parquets for ticker and compute baseline stats.

    Returns dict with keys: days, pcr_mean, pcr_std, vol_mean, vol_std.
    Today's snapshot is EXCLUDED (we compare today vs prior history).
    """
    today = dt.date.today()
    rows = []
    for f in CACHE_DIR.glob(f"{ticker}_*.parquet"):
        try:
            # Parse date from filename: TICKER_YYYY-MM-DD.parquet
            date_str = f.stem.split("_", 1)[1]
            file_date = dt.date.fromisoformat(date_str)
        except (IndexError, ValueError):
            continue
        if file_date >= today:  # exclude today
            continue
        if (today - file_date).days > lookback_days:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        call_v = int(df.loc[df["type"] == "call", "volume"].sum()) if "type" in df else 0
        put_v = int(df.loc[df["type"] == "put", "volume"].sum()) if "type" in df else 0
        if call_v == 0 and put_v == 0:
            continue
        pcr = (put_v / call_v) if call_v > 0 else 0.0
        rows.append({"date": file_date, "pcr": pcr, "total_vol": call_v + put_v})

    if not rows:
        return {"days": 0, "pcr_mean": 0, "pcr_std": 0, "vol_mean": 0, "vol_std": 0}

    hist = pd.DataFrame(rows)
    # Filter outliers in PCR (sometimes day has 1 contract = noise)
    hist = hist[(hist["pcr"] > 0) & (hist["pcr"] < 10)]
    if len(hist) < 2:
        return {
            "days": len(hist),
            "pcr_mean": float(hist["pcr"].mean()) if not hist.empty else 0,
            "pcr_std": 0, "vol_mean": float(hist["total_vol"].mean()) if not hist.empty else 0,
            "vol_std": 0,
        }
    return {
        "days": len(hist),
        "pcr_mean": float(hist["pcr"].mean()),
        "pcr_std": float(hist["pcr"].std()),
        "vol_mean": float(hist["total_vol"].mean()),
        "vol_std": float(hist["total_vol"].std()),
    }


# ---------------------------------------------------------------------------
# Smart money direction classifier
# ---------------------------------------------------------------------------
def classify_smart_money(unusual: List[UnusualContract]) -> tuple[str, float, str]:
    """Look at top unusual contracts by $ notional, weight by call/put + DTE + OTM.

    A real "smart money bet" looks like:
    - Concentrated $ flow on one side (calls OR puts)
    - Near-DTE (< 30 days) for time-sensitive event bets
    - OTM (between +1% and +25%) — speculation, not deep ITM box/arb
    - 70%+ of weighted $ flow on one side

    Returns (direction, conviction_0_100, chinese_label).
    """
    if not unusual:
        return "none", 0.0, "（無異動）"

    # Use top 8 by $ notional (the meaningful bets)
    top = sorted(unusual, key=lambda u: u.dollar_notional, reverse=True)[:8]

    def _weight(u: UnusualContract) -> float:
        """Higher weight for near-DTE + clean-OTM speculation positions."""
        w = u.dollar_notional
        # DTE multiplier: peak at DTE 7-21, decay outside
        if u.days_to_expiry <= 3:
            dte_mult = 1.3       # very near-term = high conviction event bet
        elif u.days_to_expiry <= 14:
            dte_mult = 1.5       # peak window for event bets
        elif u.days_to_expiry <= 30:
            dte_mult = 1.2
        elif u.days_to_expiry <= 45:
            dte_mult = 0.8
        else:
            dte_mult = 0.5       # > 45 days = less informative
        # OTM multiplier: clean speculation range (+1% to +25%)
        if 1 <= u.otm_pct <= 25:
            otm_mult = 1.2
        elif -2 < u.otm_pct < 1:
            otm_mult = 1.0       # ATM
        elif 25 < u.otm_pct <= 40:
            otm_mult = 0.6       # far OTM, weaker signal
        elif u.otm_pct < -2:
            otm_mult = 0.3       # ITM, often arb / rolls
        else:
            otm_mult = 0.5
        return w * dte_mult * otm_mult

    call_weight = sum(_weight(u) for u in top if u.type == "call")
    put_weight = sum(_weight(u) for u in top if u.type == "put")
    total = call_weight + put_weight

    if total <= 0:
        return "none", 0.0, "（無顯著大注）"

    call_ratio = call_weight / total
    put_ratio = put_weight / total

    # Conviction = how concentrated on one side (0=50/50, 100=100% one side)
    conviction_raw = abs(call_ratio - put_ratio) * 100

    # Threshold: 70%+ one side = clear smart money direction
    if call_ratio >= 0.70:
        # Look at top call contracts to make label more specific
        top_calls = sorted([u for u in top if u.type == "call"],
                           key=_weight, reverse=True)[:3]
        avg_dte = sum(u.days_to_expiry for u in top_calls) / len(top_calls) if top_calls else 0
        avg_otm = sum(u.otm_pct for u in top_calls) / len(top_calls) if top_calls else 0
        label = f"🟢 聰明錢睇升（{call_ratio*100:.0f}% Call加權，平均{avg_dte:.0f}日DTE，+{avg_otm:.0f}% OTM）"
        return "bullish", conviction_raw, label
    elif put_ratio >= 0.70:
        top_puts = sorted([u for u in top if u.type == "put"],
                          key=_weight, reverse=True)[:3]
        avg_dte = sum(u.days_to_expiry for u in top_puts) / len(top_puts) if top_puts else 0
        avg_otm = sum(u.otm_pct for u in top_puts) / len(top_puts) if top_puts else 0
        label = f"🔴 聰明錢睇跌（{put_ratio*100:.0f}% Put加權，平均{avg_dte:.0f}日DTE，+{avg_otm:.0f}% OTM）"
        return "bearish", conviction_raw, label
    else:
        return "mixed", conviction_raw, f"⚪ 混合（Call {call_ratio*100:.0f}% / Put {put_ratio*100:.0f}%）"


def analyze_ticker(ticker: str, save_snapshot: bool = True) -> TickerOptionsSummary:
    """Pull options chains for ticker (near expiries only) and return summary."""
    try:
        t = yf.Ticker(ticker)
        # Underlying price — use fast_info to avoid slow info()
        try:
            underlying = float(t.fast_info["last_price"])
        except Exception:
            hist = t.history(period="5d")
            underlying = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

        if underlying <= 0:
            return TickerOptionsSummary(
                ticker=ticker, underlying=0, n_contracts_scanned=0,
                call_volume=0, put_volume=0, pcr=0,
                call_oi=0, put_oi=0, pcr_oi=0,
                error="No underlying price",
            )

        expiries = _select_near_expiries(t)
        if not expiries:
            return TickerOptionsSummary(
                ticker=ticker, underlying=underlying, n_contracts_scanned=0,
                call_volume=0, put_volume=0, pcr=0,
                call_oi=0, put_oi=0, pcr_oi=0,
                error="No near-term expiries available",
            )

        all_calls, all_puts = [], []
        for ex in expiries:
            try:
                chain = t.option_chain(ex)
                calls, puts = _process_chain(ticker, ex, chain, underlying)
                if not calls.empty:
                    all_calls.append(calls)
                if not puts.empty:
                    all_puts.append(puts)
            except Exception as e:
                continue

        if not all_calls and not all_puts:
            return TickerOptionsSummary(
                ticker=ticker, underlying=underlying, n_contracts_scanned=0,
                call_volume=0, put_volume=0, pcr=0,
                call_oi=0, put_oi=0, pcr_oi=0,
                error="No chain data fetched",
            )

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        call_vol = int(calls_df["volume"].sum()) if not calls_df.empty else 0
        put_vol = int(puts_df["volume"].sum()) if not puts_df.empty else 0
        call_oi = int(calls_df["openInterest"].sum()) if not calls_df.empty else 0
        put_oi = int(puts_df["openInterest"].sum()) if not puts_df.empty else 0
        pcr = (put_vol / call_vol) if call_vol > 0 else (999.0 if put_vol > 0 else 0.0)
        pcr_oi = (put_oi / call_oi) if call_oi > 0 else 0.0

        # Find unusual contracts
        combined = pd.concat([calls_df, puts_df], ignore_index=True) if not calls_df.empty or not puts_df.empty else pd.DataFrame()
        # Apply speculation-range filters BEFORE looking for unusual signals.
        # Removes arb/box trades (deep ITM) + lottery tickets + bad pricing data.
        if not combined.empty:
            iv_ok = (combined["impliedVolatility"] >= MIN_IV) & (combined["impliedVolatility"] <= MAX_IV)
            strike_ok = (combined["otm_pct"] >= MIN_OTM_PCT) & (combined["otm_pct"] <= MAX_OTM_PCT)
            combined = combined[iv_ok & strike_ok].reset_index(drop=True)

        unusual_list: List[UnusualContract] = []

        if not combined.empty:
            # PATH 1 — "Vol/OI 異動": today's vol >> existing OI = new big position
            # opened on a contract with prior settled volume. Cleanest insider tell.
            vol_oi_mask = (
                (combined["volume"] >= MIN_CONTRACT_VOL)
                & (combined["openInterest"] >= MIN_CONTRACT_OI)
                & (combined["vol_oi"] >= VOL_OI_THRESHOLD)
            )
            vol_oi_df = combined[vol_oi_mask].sort_values("vol_oi", ascending=False).head(10)
            for _, row in vol_oi_df.iterrows():
                unusual_list.append(UnusualContract(
                    ticker=ticker, expiry=row["expiry"],
                    days_to_expiry=int(row["dte"]), type=row["type"],
                    strike=float(row["strike"]), underlying=underlying,
                    volume=int(row["volume"]), open_interest=int(row["openInterest"]),
                    vol_oi_ratio=float(row["vol_oi"]),
                    iv=float(row.get("impliedVolatility", np.nan)),
                    last_price=float(row["lastPrice"]),
                    dollar_notional=float(row["dollar_notional"]),
                    otm_pct=float(row["otm_pct"]),
                    flag="vol_oi",
                ))

            # PATH 2 — "大額流入": large $ notional even when OI=0 (yfinance OI lag,
            # or freshly opened big positions). Skip if already in Path 1.
            already = {(u.expiry, u.strike, u.type) for u in unusual_list}
            flow_mask = (
                (combined["volume"] >= LARGE_NOTIONAL_MIN_VOL)
                & (combined["dollar_notional"] >= LARGE_NOTIONAL_MIN_USD)
            )
            flow_df = combined[flow_mask].sort_values("dollar_notional", ascending=False).head(10)
            for _, row in flow_df.iterrows():
                key = (row["expiry"], float(row["strike"]), row["type"])
                if key in already:
                    continue
                unusual_list.append(UnusualContract(
                    ticker=ticker, expiry=row["expiry"],
                    days_to_expiry=int(row["dte"]), type=row["type"],
                    strike=float(row["strike"]), underlying=underlying,
                    volume=int(row["volume"]), open_interest=int(row["openInterest"]),
                    vol_oi_ratio=float(row["vol_oi"]),
                    iv=float(row.get("impliedVolatility", np.nan)),
                    last_price=float(row["lastPrice"]),
                    dollar_notional=float(row["dollar_notional"]),
                    otm_pct=float(row["otm_pct"]),
                    flag="big_flow",
                ))
                if len(unusual_list) >= 15:
                    break

        # Max Vol/OI only counts contracts that have meaningful OI (not the 0-fallback)
        if not combined.empty:
            valid_oi = combined[combined["openInterest"] >= MIN_CONTRACT_OI]
            max_vol_oi = float(valid_oi["vol_oi"].max()) if not valid_oi.empty else 0.0
        else:
            max_vol_oi = 0.0

        # Sentiment
        if pcr > 1.5:
            sentiment = "bearish"
        elif pcr < 0.4:
            sentiment = "bullish"
        else:
            sentiment = "neutral"

        summary = TickerOptionsSummary(
            ticker=ticker,
            underlying=underlying,
            n_contracts_scanned=len(combined),
            call_volume=call_vol,
            put_volume=put_vol,
            pcr=pcr,
            call_oi=call_oi,
            put_oi=put_oi,
            pcr_oi=pcr_oi,
            unusual=unusual_list,
            max_vol_oi=max_vol_oi,
            sentiment=sentiment,
        )

        # Save daily snapshot for future baseline (BEFORE computing baseline so
        # today is included in tomorrow's baseline)
        if save_snapshot and not combined.empty:
            try:
                snap_path = CACHE_DIR / f"{ticker}_{dt.date.today().isoformat()}.parquet"
                combined.to_parquet(snap_path)
            except Exception:
                pass

        # Historical baseline + z-scores (uses past snapshots, excluding today)
        baseline = compute_baseline(ticker)
        if baseline["days"] >= 3:
            summary.baseline_days = baseline["days"]
            summary.pcr_baseline = baseline["pcr_mean"]
            if baseline["pcr_std"] > 0:
                summary.pcr_zscore = (pcr - baseline["pcr_mean"]) / baseline["pcr_std"]
            summary.vol_baseline = baseline["vol_mean"]
            total_vol_today = call_vol + put_vol
            if baseline["vol_std"] > 0:
                summary.vol_zscore = (total_vol_today - baseline["vol_mean"]) / baseline["vol_std"]

        # Smart money direction (weighted by $ notional + DTE + OTM)
        direction, conviction, label = classify_smart_money(unusual_list)
        summary.smart_money = direction
        summary.smart_money_conviction = conviction
        summary.smart_money_label = label

        return summary

    except Exception as e:
        return TickerOptionsSummary(
            ticker=ticker, underlying=0, n_contracts_scanned=0,
            call_volume=0, put_volume=0, pcr=0,
            call_oi=0, put_oi=0, pcr_oi=0,
            error=str(e)[:100],
        )


def scan_options(tickers: Iterable[str], max_workers: int = 6) -> List[TickerOptionsSummary]:
    """Parallel-scan a list of tickers. Returns list of summaries.

    max_workers kept modest to avoid Yahoo throttling.
    """
    tickers = list(dict.fromkeys(tickers))
    results: List[TickerOptionsSummary] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(TickerOptionsSummary(
                    ticker=futures[fut], underlying=0, n_contracts_scanned=0,
                    call_volume=0, put_volume=0, pcr=0,
                    call_oi=0, put_oi=0, pcr_oi=0,
                    error=str(e)[:100],
                ))
    # Sort by anomaly score
    results.sort(key=lambda s: s.anomaly_score(), reverse=True)
    return results
