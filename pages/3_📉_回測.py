"""喺單一股票上回測策略表現。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.backtest import backtest, summarize
from lib.data import download_single

st.set_page_config(page_title="回測", page_icon="📉", layout="wide")
st.title("📉 回測")

# ---------- 輸入 ----------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.text_input("股票代號", value="ARM").strip().upper()
with col2:
    years = st.selectbox("回溯", [1, 2, 3, 5], index=1, format_func=lambda y: f"{y} 年")
with col3:
    run = st.button("▶ 開始回測", type="primary")

if not ticker or not run:
    st.info("揀股票同回溯年期，然後撳 **開始回測**。")
    st.stop()

lookback_days = years * 365 + 100  # 多啲warmup

with st.spinner(f"回測 {ticker}（{years}年）中…"):
    df = download_single(ticker, lookback_days=lookback_days)
    spy = download_single("SPY", lookback_days=lookback_days)

if df is None or df.empty or spy is None or spy.empty:
    st.error("拎唔到數據。")
    st.stop()

start = df.index.max() - pd.Timedelta(days=years * 365)
df_test = df.loc[df.index >= start - pd.Timedelta(days=100)]
spy_test = spy.loc[spy.index >= start - pd.Timedelta(days=100)]

trades = backtest(df_test, spy_test, ticker=ticker)
stats = summarize(trades)

# ---------- 統計 ----------
st.subheader("總結")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("交易次數", stats["trades"])
c2.metric("勝率", f"{stats['win_rate']:.1f}%")
c3.metric("平均每筆回報", f"{stats['avg_return']:+.2f}%")
c4.metric("最好", f"{stats['best']:+.2f}%")
c5.metric("最差", f"{stats['worst']:+.2f}%")
c6.metric("總回報（複利）", f"{stats['total_return']:+.2f}%")

bh_start_px = float(df_test.loc[df_test.index >= start].iloc[0]["Close"])
bh_end_px = float(df_test.iloc[-1]["Close"])
bh_ret = (bh_end_px / bh_start_px - 1) * 100
st.caption(
    f"平均持貨日數：{stats['avg_days']:.0f} 日 · "
    f"同期Buy & Hold 回報：**{bh_ret:+.2f}%**"
)

# ---------- 交易紀錄 ----------
if trades:
    st.subheader("交易明細")
    df_trades = pd.DataFrame([t.as_row() for t in trades])
    df_trades = df_trades.rename(columns={
        "Entry date": "入場日",
        "Entry $": "入場價",
        "Exit date": "離場日",
        "Exit $": "離場價",
        "Half exit $": "減半倉價",
        "Return %": "回報%",
        "Days held": "持貨日",
        "Conf at entry": "入場信心度",
        "Exit reason": "離場原因",
    })
    # 翻譯離場原因
    df_trades["離場原因"] = df_trades["離場原因"].replace({
        "Stop hit": "止蝕觸發",
        "Close < SMA50": "收市跌穿50日線",
        "End of data (still open)": "數據完（仲持貨）",
    })
    st.dataframe(df_trades, use_container_width=True, hide_index=True,
                 column_config={
                     "回報%": st.column_config.NumberColumn(format="%.2f%%"),
                     "入場價": st.column_config.NumberColumn(format="$%.2f"),
                     "離場價": st.column_config.NumberColumn(format="$%.2f"),
                     "減半倉價": st.column_config.NumberColumn(format="$%.2f"),
                 })
else:
    st.warning("呢個期間冇任何signal觸發。試吓拉長期數或者換隻股。")

# ---------- 圖表 + 交易標記 ----------
st.subheader("圖表與交易標記")
plot_df = df_test.loc[df_test.index >= start]

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="價格", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    )
)

for t in trades:
    fig.add_trace(go.Scatter(
        x=[t.entry_date], y=[t.entry_price], mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="#00e676",
                    line=dict(color="white", width=1)),
        name="入場", showlegend=False,
        hovertext=f"入場 ${t.entry_price:.2f}（信心 {t.confidence_at_entry:.0f}）",
    ))
    color = "#00e676" if t.return_pct > 0 else "#ff5252"
    fig.add_trace(go.Scatter(
        x=[t.exit_date], y=[t.exit_price], mode="markers",
        marker=dict(symbol="triangle-down", size=14, color=color,
                    line=dict(color="white", width=1)),
        name="離場", showlegend=False,
        hovertext=f"離場 ${t.exit_price:.2f}（{t.return_pct:+.2f}%）— {t.exit_reason}",
    ))
    if t.half_exit_date is not None:
        fig.add_trace(go.Scatter(
            x=[t.half_exit_date], y=[t.half_exit_price], mode="markers",
            marker=dict(symbol="diamond", size=10, color="#ffeb3b",
                        line=dict(color="white", width=1)),
            name="減半倉", showlegend=False,
            hovertext=f"減半倉 ${t.half_exit_price:.2f}（跌穿20日線）",
        ))

fig.update_layout(
    height=600,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=20, b=10),
)
fig.update_yaxes(title_text="價格 ($)")
st.plotly_chart(fig, use_container_width=True)
