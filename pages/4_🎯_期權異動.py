"""期權異動掃描 — 揾市場異常買call/put嘅股。

掃近月（60日內到期）期權，計：
- PCR（Put/Call Volume Ratio）— 整體看好/看淡傾斜
- Vol/OI 比率 — 邊個strike有「新開大倉」嘅pattern
- $ Notional flow — 邊度有大錢進入
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from lib.options import HOT_TICKERS, scan_options

st.set_page_config(page_title="期權異動", page_icon="🎯", layout="wide")
st.title("🎯 期權異動掃描")

# ---------- 側邊欄 ----------
with st.sidebar:
    st.header("掃描設定")
    mode = st.radio(
        "掃邊個名單",
        ["Hot list（活躍期權~120隻）", "自訂股票"],
        index=0,
    )
    custom_list = st.text_area(
        "自訂股票（每行一隻）",
        height=120,
        value="FUTU\nNVDA\nTSLA\nAAPL",
        placeholder="FUTU\nNVDA\nTSLA",
    )
    min_score = st.slider("最低異動分數", 0, 100, 25, 5)
    rescan = st.button("🔄 重新掃描", help="清cache再拎最新數據")

    with st.expander("📖 點睇個分數"):
        st.markdown(
            """
**異動分數（0-100）**：

- **PCR > 1.5**：put成交多過call → 看淡傾斜（或人哋buy protection）
- **PCR < 0.4**：call多過put → 看好傾斜
- **Vol/OI ≥ 2**：今日成交係之前OI嘅2倍以上 = 新開大倉 = 高度可疑
- **$ Notional**：每張合約成交額（vol × 價 × 100）

⚠️ yfinance對活躍股嘅OI經常返0（data lag），所以Vol/OI route有時唔fire。
但**PCR + $ flow**已經夠sharp揾出insider activity。
            """
        )


# ---------- 掃描 ----------
if mode == "Hot list（活躍期權~120隻）":
    tickers = HOT_TICKERS
else:
    tickers = [
        t.strip().upper()
        for t in (custom_list or "").splitlines()
        if t.strip()
    ]

if not tickers:
    st.info("請輸入至少一隻股票。")
    st.stop()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_scan(tickers_tuple: tuple) -> list:
    return scan_options(list(tickers_tuple))


if rescan:
    st.cache_data.clear()

n = len(tickers)
est_time = max(15, n // 3)  # ~3 tickers/sec with threading
with st.spinner(f"掃描 {n} 隻股票嘅期權鏈…（預計~{est_time}秒）"):
    results = cached_scan(tuple(tickers))

# ---------- 顯示 ----------
ok_results = [r for r in results if not r.error]
err_results = [r for r in results if r.error]

st.caption(f"掃完 {len(ok_results)} 隻 · 失敗 {len(err_results)} · {sum(r.n_contracts_scanned for r in ok_results):,} 張合約已分析")

filtered = [r for r in ok_results if r.anomaly_score() >= min_score]

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("總異動股", len(filtered))
m2.metric("看淡傾斜", sum(1 for r in filtered if r.sentiment == "bearish"))
m3.metric("看好傾斜", sum(1 for r in filtered if r.sentiment == "bullish"))
total_notional = sum(sum(u.dollar_notional for u in r.unusual) for r in filtered)
m4.metric("總異動$ flow", f"${total_notional/1_000_000:.1f}M")


def build_summary_df(results_list):
    rows = []
    for r in results_list:
        top_unusual = r.unusual[0] if r.unusual else None
        # Find biggest call & biggest put by $ notional
        biggest_call = max(
            (u for u in r.unusual if u.type == "call"),
            key=lambda u: u.dollar_notional, default=None,
        )
        biggest_put = max(
            (u for u in r.unusual if u.type == "put"),
            key=lambda u: u.dollar_notional, default=None,
        )
        rows.append({
            "股票": r.ticker,
            "現價": round(r.underlying, 2),
            "異動分數": round(r.anomaly_score(), 1),
            "PCR": round(r.pcr, 2),
            "傾斜": {"bullish": "🟢 看好", "bearish": "🔴 看淡", "neutral": "⚪ 中性"}[r.sentiment],
            "Call$": int(sum(u.dollar_notional for u in r.unusual if u.type == "call")),
            "Put$": int(sum(u.dollar_notional for u in r.unusual if u.type == "put")),
            "最大Call": (
                f"${biggest_call.strike:.0f} ({biggest_call.days_to_expiry}d)"
                if biggest_call else "—"
            ),
            "最大Put": (
                f"${biggest_put.strike:.0f} ({biggest_put.days_to_expiry}d)"
                if biggest_put else "—"
            ),
            "Max Vol/OI": round(r.max_vol_oi, 1) if r.max_vol_oi > 0 else None,
            "#異動": len(r.unusual),
        })
    return pd.DataFrame(rows)


tab_all, tab_bear, tab_bull = st.tabs(
    [
        f"📋 全部排名 ({len(filtered)})",
        f"🔴 看淡異動 ({sum(1 for r in filtered if r.sentiment == 'bearish')})",
        f"🟢 看好異動 ({sum(1 for r in filtered if r.sentiment == 'bullish')})",
    ]
)


def render_table(tab, items, blurb):
    with tab:
        st.markdown(blurb)
        if not items:
            st.info("無符合條件嘅股票。")
            return
        df = build_summary_df(items)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "異動分數": st.column_config.ProgressColumn(
                    "異動分數", min_value=0, max_value=100, format="%.1f"
                ),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "Call$": st.column_config.NumberColumn("Call $流入", format="$%d"),
                "Put$": st.column_config.NumberColumn("Put $流入", format="$%d"),
                "PCR": st.column_config.NumberColumn(
                    "PCR",
                    help=">1.5 看淡 · 0.4-1.5 中性 · <0.4 看好",
                    format="%.2f",
                ),
                "Max Vol/OI": st.column_config.NumberColumn(
                    help="今日成交 / 之前未平倉 — >2 有新開大倉嫌疑",
                    format="%.1fx",
                ),
            },
        )


render_table(
    tab_all,
    sorted(filtered, key=lambda r: r.anomaly_score(), reverse=True),
    "**全部異動股按分數排序**。click落面drill-down睇具體合約。",
)
render_table(
    tab_bear,
    sorted([r for r in filtered if r.sentiment == "bearish"], key=lambda r: r.anomaly_score(), reverse=True),
    "**🔴 看淡異動** — Put成交明顯多過Call。可能：(a) 對沖盤 (b) 有人賭跌 (c) insider知壞消息。Drill-down睇邊個strike + DTE 揾pattern。",
)
render_table(
    tab_bull,
    sorted([r for r in filtered if r.sentiment == "bullish"], key=lambda r: r.anomaly_score(), reverse=True),
    "**🟢 看好異動** — Call成交多過Put。可能：(a) 有人賭升 (b) 公司有好消息 (c) earnings/M&A傳聞。",
)

# ---------- Drill-down per ticker ----------
st.markdown("---")
st.subheader("🔍 個別股票drill-down")
ticker_choice = st.selectbox(
    "揀股票睇晒所有unusual contracts",
    [r.ticker for r in filtered] if filtered else ["（無）"],
    index=0,
)

if filtered and ticker_choice != "（無）":
    r = next(r for r in filtered if r.ticker == ticker_choice)
    st.markdown(
        f"### {r.ticker} · 現價 ${r.underlying:.2f} · "
        f"PCR **{r.pcr:.2f}** ({r.sentiment}) · 異動分數 **{r.anomaly_score():.1f}**"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Call 成交", f"{r.call_volume:,}")
    col_b.metric("Put 成交", f"{r.put_volume:,}")
    col_c.metric("掃描張數", f"{r.n_contracts_scanned:,}")

    if r.unusual:
        df_u = pd.DataFrame([u.as_row() for u in r.unusual])
        # Color-code calls vs puts
        st.dataframe(
            df_u,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="$%.2f"),
                "Vol": st.column_config.NumberColumn(format="%d"),
                "OI": st.column_config.NumberColumn(format="%d"),
                "Vol/OI": st.column_config.NumberColumn(format="%.1fx"),
                "IV": st.column_config.NumberColumn(format="%.0f%%"),
                "OTM%": st.column_config.NumberColumn(
                    help="正數 = OTM (超出價內) · 負數 = ITM (深入價內)",
                    format="%.1f%%",
                ),
                "$ Notional": st.column_config.NumberColumn(format="$%d"),
                "DTE": st.column_config.NumberColumn(help="距離到期日數"),
            },
        )
        st.caption(
            "**Signal欄解釋**：「Vol/OI異動」= 今日成交多過之前未平倉（新開大倉）。"
            "「大額流入」= 合約$流入超過$200K（即使OI=0都當顯著流入）。"
        )
    else:
        st.info("呢隻股冇符合異動條件嘅合約。")

if err_results:
    with st.expander(f"⚠️ {len(err_results)} 隻拎唔到數據"):
        st.write([f"{r.ticker}: {r.error}" for r in err_results])
