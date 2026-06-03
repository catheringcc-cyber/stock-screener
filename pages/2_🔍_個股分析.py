"""個股分析：K線、EMA指標、入場/止蝕/目標、跟價止賺建議。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib.data import download_prices, download_single
from lib.indicators import enrich
from lib.minervini import CONDITION_LABELS, check_trend_template, compute_rs_ratings
from lib.options import analyze_ticker
from lib.strategy import evaluate
from lib.universe import get_universe

st.set_page_config(page_title="個股分析", page_icon="🔍", layout="wide")
st.title("🔍 個股分析")

# ---------- 輸入 ----------
col_inp, col_period = st.columns([2, 1])
with col_inp:
    ticker = st.text_input("股票代號", value="ARM").strip().upper()
with col_period:
    months_back = st.selectbox("睇返", [3, 6, 12, 18], index=2, format_func=lambda m: f"{m} 個月")

if not ticker:
    st.stop()

# ---------- 載入數據 ----------
with st.spinner(f"載入 {ticker}…"):
    df_raw = download_single(ticker)
    spy_raw = download_single("SPY")

if df_raw is None or df_raw.empty:
    st.error(f"揾唔到 {ticker} 嘅數據")
    st.stop()

df = enrich(df_raw).iloc[-(months_back * 21):]
sig = evaluate(ticker, df_raw, spy_raw)

# ---------- 頂部指標列（用EMA作主指標） ----------
last = df.iloc[-1]
prev = df.iloc[-2]
chg = last["Close"] - prev["Close"]
chg_pct = chg / prev["Close"] * 100
ema20_slope = last.get("EMA20_slope10")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("現價", f"${last['Close']:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
m2.metric(
    "EMA20",
    f"${last['EMA20']:.2f}" if pd.notna(last["EMA20"]) else "—",
    f"{(last['Close']/last['EMA20']-1)*100:+.1f}% 距離" if pd.notna(last["EMA20"]) else None,
)
# 斜率係策略最重要單一指標，特別highlight
if pd.notna(ema20_slope):
    if 3 <= ema20_slope <= 7:
        slope_label = "🟢 甜蜜區"
    elif ema20_slope > 7:
        slope_label = "🟠 偏陡" if ema20_slope <= 15 else "🔴 太parabolic"
    elif ema20_slope > 0:
        slope_label = "🟡 偏弱" if ema20_slope < 2 else "🟢 正常"
    else:
        slope_label = "🔴 向下"
    m3.metric("EMA20 斜率（10日）", f"{ema20_slope:+.2f}%", slope_label)
else:
    m3.metric("EMA20 斜率（10日）", "—")

m4.metric("EMA50", f"${last['EMA50']:.2f}" if pd.notna(last["EMA50"]) else "—")
m5.metric("RSI(14)", f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
m6.metric(
    "波幅百分位",
    f"{last['HV_pct']:.0f}" if pd.notna(last.get("HV_pct")) else "—",
    "option可能平" if pd.notna(last.get("HV_pct")) and last["HV_pct"] < 30
    else ("option偏貴" if pd.notna(last.get("HV_pct")) and last["HV_pct"] > 70 else "中性"),
    delta_color="off",
)

st.caption(
    "💡 **EMA20斜率係策略最關鍵指標**：3-7%/10日 = 45°最強訊號 · "
    ">15% = parabolic要避 · <0 = 趨勢未起。"
)

# ---------- 訊號狀態 ----------
if sig is None:
    st.warning(
        "**唔喺任何bucket。** 呢隻股未達標 — 可能趨勢未起，又或者趨勢已破。"
    )
else:
    color = {"entry": "🟢", "pullback": "🟡", "strong": "🔵", "emerging": "🌱"}[sig.bucket]
    bucket_zh = {
        "entry": "入場", "pullback": "回調",
        "strong": "強勢", "emerging": "早期轉強",
    }[sig.bucket]
    st.success(
        f"{color} **{bucket_zh}** · 信心度 **{sig.confidence:.1f}/100** · {sig.notes}"
    )
    s = sig.stage_scores
    st.caption(
        f"各階段分數 → 早期轉強: {s.get('emerging','—')} · "
        f"強勢: {s['strong']} · 回調: {s['pullback']} · 入場: {s['entry']}"
    )

# ---------- Minervini Trend Template ----------
st.markdown("---")
st.subheader("🏆 Minervini Trend Template 檢查")
st.caption(
    "Minervini嘅 8 個gatekeeper條件 — **缺一不可**。出自《Think & Trade Like a Champion》。"
)


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_rs_for_ticker(ticker: str) -> float | None:
    """Compute RS rating for this ticker relative to S&P 500 + NDX universe."""
    universe = get_universe("sp500_ndx")
    if ticker not in universe:
        universe = list(universe) + [ticker]
    data = download_prices(universe)
    ratings = compute_rs_ratings(data)
    return ratings.get(ticker)


with st.spinner("計算 RS Rating（同股池內所有股票比較）…"):
    rs_rating = _compute_rs_for_ticker(ticker)

mv_result = check_trend_template(enrich(df_raw), rs_rating=rs_rating)

# Headline
mv_color = "✅" if mv_result.all_pass else ("⭐" if mv_result.passed >= 7 else "⚠️" if mv_result.passed >= 5 else "❌")
st.markdown(
    f"### {mv_color} **{mv_result.passed}/8 通過** · "
    f"RS Rating = **{rs_rating:.0f}**" if rs_rating is not None else f"### {mv_color} **{mv_result.passed}/8 通過** · RS Rating: 不適用（歷史不足252日）"
)

# Conditions checklist (2 columns)
col_l, col_r = st.columns(2)
for i in range(1, 9):
    passed = mv_result.conditions.get(i, False)
    icon = "✅" if passed else "❌"
    label = CONDITION_LABELS[i]

    # Add explanatory subtext for each condition
    subtext = ""
    if i == 1 and mv_result.sma150 is not None:
        subtext = f"現價 ${mv_result.price:.2f} {'>' if passed else '≤'} SMA150 ${mv_result.sma150:.2f}"
    elif i == 2 and mv_result.sma200 is not None:
        subtext = f"現價 ${mv_result.price:.2f} {'>' if passed else '≤'} SMA200 ${mv_result.sma200:.2f}"
    elif i == 3 and mv_result.sma150 is not None and mv_result.sma200 is not None:
        subtext = f"SMA150 ${mv_result.sma150:.2f} {'>' if passed else '≤'} SMA200 ${mv_result.sma200:.2f}"
    elif i == 4 and mv_result.sma200 is not None and mv_result.sma200_21d_ago is not None:
        change_pct = (mv_result.sma200 / mv_result.sma200_21d_ago - 1) * 100
        subtext = f"SMA200 21日變化 {change_pct:+.2f}%"
    elif i == 5 and mv_result.sma50 is not None:
        subtext = f"SMA50 ${mv_result.sma50:.2f}"
    elif i == 6 and mv_result.pct_from_52w_low is not None:
        subtext = f"離52週低 +{mv_result.pct_from_52w_low:.1f}%（需要 ≥30%）"
    elif i == 7 and mv_result.pct_below_52w_high is not None:
        subtext = f"距52週高 -{mv_result.pct_below_52w_high:.1f}%（需要 ≤25%）"
    elif i == 8:
        subtext = f"RS Rating = {rs_rating:.0f}" if rs_rating is not None else "RS Rating 無法計算"

    target_col = col_l if i <= 4 else col_r
    with target_col:
        st.markdown(f"**{icon} 條件 {i}**：{label}")
        if subtext:
            st.caption(subtext)

# Summary 52w stats
if mv_result.week52_high is not None and mv_result.week52_low is not None:
    st.markdown("**52週統計**")
    w_cols = st.columns(3)
    w_cols[0].metric("52週低", f"${mv_result.week52_low:.2f}",
                     f"+{mv_result.pct_from_52w_low:.1f}% 離底" if mv_result.pct_from_52w_low else None)
    w_cols[1].metric("52週高", f"${mv_result.week52_high:.2f}",
                     f"-{mv_result.pct_below_52w_high:.1f}% 距頂" if mv_result.pct_below_52w_high else None,
                     delta_color="inverse")
    w_cols[2].metric("現價位置",
                     f"{(mv_result.price - mv_result.week52_low) / (mv_result.week52_high - mv_result.week52_low) * 100:.0f}%",
                     "0% = 52週低, 100% = 52週高")

if mv_result.all_pass:
    st.success("🏆 **完美通過 8/8**。Trend Template綠燈 — 可以進一步睇VCP形態 + Entry signal。")
elif mv_result.passed == 7:
    st.warning(f"⭐ **過咗 7/8**。Minervini本人話過：「過7條都唔夠」。唯一冇過嘅係條件 "
               f"{[i for i, v in mv_result.conditions.items() if not v]}")
else:
    failed_list = [f"{i}." + CONDITION_LABELS[i] for i, v in mv_result.conditions.items() if not v]
    st.info("❌ **未過Trend Template**。冇過：\n- " + "\n- ".join(failed_list))


# ---------- 交易水平 ----------
st.subheader("交易水平")
if sig is not None:
    entry, stop, t1, t2 = sig.entry, sig.stop, sig.target, sig.target2
    risk_pct = (entry - stop) / entry * 100
    reward_pct = (t1 - entry) / entry * 100

    lvl_cols = st.columns(4)
    lvl_cols[0].metric(
        "入場價", f"${entry:.2f}",
        "市價入" if sig.bucket in ("entry", "emerging") else "突破入",
    )
    lvl_cols[1].metric("止蝕", f"${stop:.2f}", f"-{risk_pct:.2f}%", delta_color="inverse")
    lvl_cols[2].metric("目標1（2R）", f"${t1:.2f}", f"+{reward_pct:.2f}%")
    lvl_cols[3].metric("目標2（量度升幅）", f"${t2:.2f}",
                       f"+{(t2-entry)/entry*100:.2f}%")

# 持貨中嘅跟價止賺
st.markdown("**如果你已經持貨**，跟價止賺邏輯建議：")
tr_cols = st.columns(3)
ema20_now = float(last["EMA20"]) if pd.notna(last["EMA20"]) else None
ema50_now = float(last["EMA50"]) if pd.notna(last["EMA50"]) else None
above_20 = ema20_now is not None and last["Close"] > ema20_now
tr_cols[0].metric("跟價止賺（趨勢未死）", f"${ema20_now:.2f}" if ema20_now else "—",
                  "EMA20 — 跌穿減半倉")
tr_cols[1].metric("最後防線", f"${ema50_now:.2f}" if ema50_now else "—",
                  "EMA50 — 跌穿全走")
tr_cols[2].metric(
    "趨勢狀態",
    "✅ 企EMA20之上" if above_20 else "⚠️ 跌穿EMA20",
    "繼續持有" if above_20 else "考慮減倉",
)

# ---------- 圖表 ----------
st.subheader("圖表")
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.15, 0.25],
    vertical_spacing=0.03,
    subplot_titles=("價格 + EMA", "成交量", "RSI(14)"),
)

# K線
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="價格", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
)
# EMA（主指標）
for col, color, label, width in [
    ("EMA5", "#f9a825", "EMA5（快線）", 1.2),
    ("EMA20", "#42a5f5", "EMA20（主趨勢）", 2.2),
    ("EMA50", "#ab47bc", "EMA50", 1.8),
]:
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col], name=label, line=dict(width=width, color=color),
                   hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>"),
        row=1, col=1,
    )

# 交易水平
if sig is not None:
    for level, label, color in [
        (sig.entry, f"入場 ${sig.entry:.2f}", "#ffffff"),
        (sig.stop, f"止蝕 ${sig.stop:.2f}", "#ef5350"),
        (sig.target, f"目標1 ${sig.target:.2f}", "#26a69a"),
        (sig.target2, f"目標2 ${sig.target2:.2f}", "#66bb6a"),
    ]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10, row=1, col=1)

# 成交量
vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(
    go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=vol_colors,
           showlegend=False),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["VolAvg20"], name="20日均量",
               line=dict(width=1, color="#ffeb3b"), showlegend=False),
    row=2, col=1,
)

# RSI
fig.add_trace(
    go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)",
               line=dict(color="#ab47bc", width=1.5), showlegend=False),
    row=3, col=1,
)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,82,82,0.5)", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=3, col=1)
fig.add_hrect(y0=40, y1=60, fillcolor="rgba(126,87,194,0.15)", line_width=0,
              annotation_text="入場區", annotation_position="top left",
              annotation_font_size=10, row=3, col=1)

fig.update_layout(
    height=750,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_yaxes(title_text="價格 ($)", row=1, col=1)
fig.update_yaxes(title_text="量", row=2, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------- 期權異動 ----------
st.markdown("---")
st.subheader("🎯 期權異動")
st.caption(
    "近60日到期合約嘅PCR + 異常成交。Insider/大戶通常買近月合約搏specific event。"
)

with st.spinner(f"拎 {ticker} 期權數據…"):
    opt = analyze_ticker(ticker, save_snapshot=False)

if opt.error:
    st.warning(f"拎唔到期權數據：{opt.error}")
else:
    o1, o2, o3, o4 = st.columns(4)
    o1.metric(
        "PCR（成交比）",
        f"{opt.pcr:.2f}",
        {"bullish": "🟢 看好傾斜", "bearish": "🔴 看淡傾斜", "neutral": "⚪ 中性"}[opt.sentiment],
        delta_color="off",
    )
    o2.metric("Call 成交", f"{opt.call_volume:,}")
    o3.metric("Put 成交", f"{opt.put_volume:,}")
    o4.metric("異動分數", f"{opt.anomaly_score():.1f}/100")

    if opt.unusual:
        st.markdown(f"**Top {len(opt.unusual)} unusual contracts**（按signal排序）：")
        df_opt = pd.DataFrame([u.as_row() for u in opt.unusual])
        st.dataframe(
            df_opt,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="$%.2f"),
                "Vol": st.column_config.NumberColumn(format="%d"),
                "OI": st.column_config.NumberColumn(format="%d"),
                "Vol/OI": st.column_config.NumberColumn(format="%.1fx"),
                "IV": st.column_config.NumberColumn(format="%.0f%%"),
                "OTM%": st.column_config.NumberColumn(format="%.1f%%"),
                "$ Notional": st.column_config.NumberColumn(format="$%d"),
            },
        )
    else:
        st.info("呢隻股無達標嘅異動合約。可能流量正常或者yfinance OI數據空白。")

    st.caption(
        "💡 **點解讀**：\n"
        "- 大量put集中喺特定OTM strike + 近DTE → 有人賭跌（insider tell？）\n"
        "- 大量call集中喺ATM/輕微OTM + 中期DTE → 有人賭升（earnings beat / M&A）\n"
        "- Vol/OI > 2 = 今日新開大倉。OI = 0 但$ notional高 = 大資金壓注（OI lag）。"
    )
