"""每日選股 — 4 bucket分類，按actionability排序。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from lib.data import download_prices
from lib.strategy import scan_universe
from lib.universe import get_universe

st.set_page_config(page_title="Screener", page_icon="📊", layout="wide")
st.title("📊 Screener 選股")

# ---------- 側邊欄 ----------
with st.sidebar:
    st.header("掃描設定")
    universe_choice = st.selectbox(
        "股池",
        ["fallback", "sp500", "sp500_ndx"],
        format_func={
            "fallback": "精選動量股（~99隻，快）",
            "sp500": "S&P 500（500隻，慢）",
            "sp500_ndx": "S&P 500 + Nasdaq 100（~517隻，最慢）",
        }.get,
        index=0,
        help="細名單載入快、慳記憶體。S&P 500+NDX 覆蓋多但首次掃要1-2分鐘 + Cloud免費tier得1GB RAM可能OOM。",
    )

    custom_list = st.text_area(
        "自訂股票（每行一隻，會覆蓋股池）",
        height=100,
        placeholder="AAPL\nNVDA\nARM",
    )

    min_conf = st.slider("最低信心度", 0, 100, 50, 5)
    max_results = st.slider("每個bucket最多顯示", 5, 40, 20, 5)
    minervini_only = st.checkbox(
        "🏆 只睇Minervini 8/8 通過嘅股",
        value=False,
        help="開咗會喺所有bucket只顯示通過晒8條Trend Template嘅股票",
    )

    rescan = st.button("🔄 強制重掃", help="清cache、重新由Yahoo拎數據")


# ---------- 掃描 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def run_scan(universe_key: str, custom: tuple[str, ...]):
    if custom:
        tickers = list(custom) + ["SPY"]
    else:
        tickers = get_universe(universe_key)
    data = download_prices(tickers)
    signals = scan_universe(data)
    return signals, len(data)


if rescan:
    st.cache_data.clear()
    cache_dir = Path(__file__).parent.parent / ".cache" / "prices"
    today = pd.Timestamp.utcnow().date().isoformat()
    for f in cache_dir.glob(f"*_{today}.parquet"):
        f.unlink()

custom_tuple = tuple(
    t.strip().upper() for t in (custom_list or "").splitlines() if t.strip()
)

with st.spinner("掃描中…（首次掃整個股池要1-2分鐘，之後讀cache好快）"):
    signals, n_loaded = run_scan(universe_choice, custom_tuple)

n_minervini = sum(1 for s in signals if s.minervini and s.minervini.all_pass)
n_vcp_ready = sum(1 for s in signals if s.vcp and "Ready" in s.vcp.maturity)
n_vcp_forming = sum(1 for s in signals if s.vcp and "Forming" in s.vcp.maturity)
n_vcp_breakout = sum(1 for s in signals if s.vcp and "Broken" in s.vcp.maturity)
st.caption(
    f"已載入 {n_loaded} 隻 · 揾到 {len(signals)} 個signals · "
    f"🏆 **{n_minervini}** 隻通過Minervini Trend Template 8/8 · "
    f"VCP 🟢 Ready **{n_vcp_ready}** · 🟡 Forming **{n_vcp_forming}** · 🚀 突破 **{n_vcp_breakout}**"
)

# ---------- 篩選同顯示 ----------
filtered = [s for s in signals if s.confidence >= min_conf]
if minervini_only:
    filtered = [s for s in filtered if s.minervini and s.minervini.all_pass]

tab_vcp, tab_minervini, tab_entry, tab_pullback, tab_strong, tab_emerging = st.tabs(
    [
        f"🎯 VCP Setup ({sum(1 for s in filtered if s.vcp and ('Ready' in s.vcp.maturity or 'Forming' in s.vcp.maturity or 'Broken' in s.vcp.maturity))})",
        f"🏆 Minervini 8/8 ({sum(1 for s in filtered if s.minervini and s.minervini.all_pass)})",
        f"🎯 入場訊號 ({sum(1 for s in filtered if s.bucket == 'entry')})",
        f"⏳ 回調中 ({sum(1 for s in filtered if s.bucket == 'pullback')})",
        f"💪 強勢觀察 ({sum(1 for s in filtered if s.bucket == 'strong')})",
        f"🌱 轉強候選 ({sum(1 for s in filtered if s.bucket == 'emerging')})",
    ]
)


def _render_table(items_list, max_n):
    rows = [s.as_row() for s in items_list][:max_n]
    if not rows:
        st.info("依家無股符合呢個信心度。")
        return
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "Ticker": "股票",
        "Bucket": "Bucket",
        "Confidence": "信心度",
        "Minervini": "Minervini",
        "RS": "RS Rating",
        "VCP": "VCP",
        "VCP tight%": "VCP收窄%",
        "距Pivot%": "距Pivot%",
        "Pivot $": "Pivot $",
        "Price": "現價",
        "Entry": "入場價",
        "Stop": "止蝕",
        "T1": "目標1",
        "T2": "目標2",
        "R:R (T1)": "回報/風險",
        "Stop %": "止蝕%",
        "EMA20 slope %": "EMA20斜率%",
        "HV pct": "波幅百分位",
        "Notes": "備註",
    })
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "信心度": st.column_config.ProgressColumn(
                "信心度", min_value=0, max_value=100, format="%.1f"
            ),
            "Minervini": st.column_config.TextColumn(
                "Minervini",
                help="Minervini Trend Template 8條件通過幾多。🏆 8/8 = 完美 · ⭐ 7/8 · ⚠️ 5-6/8 · ❌ ≤4/8",
            ),
            "RS Rating": st.column_config.NumberColumn(
                "RS Rating",
                help="IBD式 12個月relative strength percentile rank（1-99）。≥70為Minervini第8條，理想85+。",
                format="%.0f",
            ),
            "VCP": st.column_config.TextColumn(
                "VCP",
                help="VCP setup成熟度：🟢 Ready（≥3T+volume遞減+tight<3%+距pivot<3%）· 🟡 Forming（≥2T+volume遞減+喺base內）· 🚀 已突破 · ⚪ 無VCP",
            ),
            "VCP收窄%": st.column_config.NumberColumn(
                "VCP收窄%",
                help="最近12日平均daily range / 股價。<3% = tight base, ≥5% = 仲未coil好。",
                format="%.2f%%",
            ),
            "距Pivot%": st.column_config.NumberColumn(
                "距Pivot%",
                help="現價距離VCP pivot(阻力)的距離。負數=喺pivot之下；0% = 啱啱touching；正數 = 已突破。",
                format="%.2f%%",
            ),
            "Pivot $": st.column_config.NumberColumn(
                "Pivot $",
                help="VCP最後contraction嘅高位。突破呢個價位 = buy signal。",
                format="$%.2f",
            ),
            "EMA20斜率%": st.column_config.NumberColumn(
                "EMA20斜率%",
                help="EMA20過去10日%變化。3-7%係45°甜蜜區，>15%太parabolic。",
                format="%.2f%%",
            ),
            "波幅百分位": st.column_config.NumberColumn(
                "波幅百分位",
                help="HV percentile，越低越平靜（option可能平），越高越貴。",
                format="%.0f",
            ),
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "入場價": st.column_config.NumberColumn(format="$%.2f"),
            "止蝕": st.column_config.NumberColumn(format="$%.2f"),
            "目標1": st.column_config.NumberColumn(format="$%.2f"),
            "目標2": st.column_config.NumberColumn(format="$%.2f"),
            "止蝕%": st.column_config.NumberColumn(format="%.2f%%"),
            "回報/風險": st.column_config.NumberColumn(format="%.2fx"),
        },
    )
    st.caption(
        "目標1 = 入場 + 2×風險 · 目標2 = 量度升幅目標 · 止蝕喺[-7%, -3%]之間"
    )


def render_bucket(tab, bucket_name: str, blurb: str):
    items = [s for s in filtered if s.bucket == bucket_name]
    with tab:
        st.markdown(blurb)
        _render_table(items, max_results)


# VCP Setup tab — show Ready + Forming + Broken out across all buckets
with tab_vcp:
    st.markdown(
        "**🎯 VCP setup detected**：呢度顯示有清晰VCP pattern嘅股票。"
        "Pattern越成熟（🟢 Ready），距pivot越近 → 突破時嘅conviction越高。"
    )
    vcp_stocks = [s for s in filtered if s.vcp and s.vcp.maturity != "⚪ Not yet"]
    # Sort: Ready first, then Forming, then Broken out; within each by RS desc
    def _vcp_sort(s):
        order = {"🟢 Ready": 0, "🟡 Forming": 1, "🚀 Broken out": 2}.get(s.vcp.maturity, 9)
        rs = s.minervini.rs_rating if s.minervini and s.minervini.rs_rating is not None else 0
        return (order, -rs)
    vcp_stocks.sort(key=_vcp_sort)
    _render_table(vcp_stocks, max_results)
    if vcp_stocks:
        n_r = sum(1 for s in vcp_stocks if "Ready" in s.vcp.maturity)
        n_f = sum(1 for s in vcp_stocks if "Forming" in s.vcp.maturity)
        n_b = sum(1 for s in vcp_stocks if "Broken" in s.vcp.maturity)
        st.caption(
            f"🟢 Ready: {n_r}（即刻可以set price alert喺pivot）· "
            f"🟡 Forming: {n_f}（base未夠tight，繼續觀察）· "
            f"🚀 突破: {n_b}（已經破pivot，可以追入或者等回踩）"
        )


# Minervini 8/8 tab — cross-bucket view of the gatekeeper winners
with tab_minervini:
    st.markdown(
        "**🏆 Minervini Trend Template 全部 8 條件通過。** 呢個係swing trade嘅gate keeper — "
        "Minervini本人講過「過7條都唔夠」。喺呢個list入面再揀bucket睇，係最高conviction嘅setup。"
    )
    perfect = [s for s in filtered if s.minervini and s.minervini.all_pass]
    # Sort by RS Rating desc within the tab
    perfect.sort(key=lambda s: -(s.minervini.rs_rating or 0))
    _render_table(perfect, max_results)
    if perfect:
        st.caption(
            f"已按RS Rating由高到低排。最高分stock RS = {perfect[0].minervini.rs_rating:.0f}。"
        )

render_bucket(
    tab_entry,
    "entry",
    "**今日可入場。** 3階段全部確認 — 趨勢強、有回調、確認陽燭出。市價入，即刻set定止蝕。",
)
render_bucket(
    tab_pullback,
    "pullback",
    "**密切觀察。** 強勢股回到EMA20附近、成交縮量。喺入場價set price alert — "
    "等陽燭企返EMA20同有量先入。**特別留意「斜率高+距離近」嘅黃金位**。",
)
render_bucket(
    tab_strong,
    "strong",
    "**加入watchlist。** EMA20斜率正常、跑贏大市，但未回調。等 — 追高係穩死。"
    " 部分斜率>15%已經接近parabolic，要更加耐性等。",
)
render_bucket(
    tab_emerging,
    "emerging",
    "**🌱 早期候選。** EMA20剛由走平翹上、EMA5啱啱金叉EMA20、成交有量。"
    "呢類股可能2-4星期後變強勢。**風險高啲，但upside最大** — "
    "可以小注試水，或者set price alert等突破。",
)
