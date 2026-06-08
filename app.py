"""動量 + 回調選股 — 首頁。

啟動方法：
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="動量選股",
    page_icon="📈",
    layout="wide",
)

st.title("📈 動量 + 回調選股")

st.success(
    "🆕 **最新加入**：🎯 **VCP (Volatility Contraction Pattern) 檢測 + ATR auto-threshold**　·　"
    "🏆 Minervini Trend Template 8條件 gate keeper　·　"
    "🎯 期權異動掃描（PCR + 大額$流入 + 聰明錢方向）"
)

st.markdown(
    """
### 4 個 page，4 個工具

| Page | 做咩 |
|---|---|
| **📊 選股** | 每日掃 S&P 500 + Nasdaq 100，分 **6 個 tab**：🎯 **VCP Setup**、🏆 Minervini 8/8、🎯 入場、⏳ 回調、💪 強勢、🌱 轉強候選 |
| **🔍 個股分析** | K線 + EMA5/20/50 + RSI + **Minervini 8條件 checklist** + 52週位置 + **🎯 VCP 分析（contraction series + pivot line + zigzag overlay）** + 入場/止蝕/目標 + **期權異動drill-down** |
| **📉 回測** | Walk-forward 回測單一股票嘅策略表現 |
| **🎯 期權異動** | 掃 ~120 隻活躍options名，揾 **PCR異常 + 大額$流入 + 聰明錢方向 + Vol/OI異動** |

---

### 策略邏輯（三層 gate keeper + 4 個入場階段）

**Layer 1 — Minervini Trend Template（8條件全部要過）**

呢個係swing trade嘅基本門檻。Minervini本人講過：「過 7 條都唔夠」。

| # | 條件 |
|---|---|
| 1 | 股價 > SMA150 |
| 2 | 股價 > SMA200 |
| 3 | SMA150 > SMA200 |
| 4 | SMA200 向上至少 1 個月 |
| 5 | SMA50 > SMA150/200，股價 > SMA50 |
| 6 | 股價離 52週低位 ≥ 30% |
| 7 | 股價喺 52週高位 25% 範圍內 |
| 8 | RS Rating ≥ 70 |

**Layer 2 — VCP（Volatility Contraction Pattern）setup**

過咗Minervini之後，睇VCP成熟度。VCP = 一連串越嚟越淺嘅回調 + 成交量遞減，最後收窄到 pivot：

| Maturity | 條件 | 動作 |
|---|---|---|
| 🟢 **Ready** | ≥2 contractions、tightness <5%、距 pivot -3% 至 +2% | **set price alert喺 pivot 等突破** |
| 🟡 **Forming** | ≥2 contractions + 量遞減、距 pivot -10% 至 +2% | 觀察、等再收窄 |
| 🚀 **Broken out** | 已突破 pivot 0-15% | 追入或等回踩 |

Zigzag threshold用 ATR-based 自動調整（1× ATR(20) / 股價），$1700 股票 vs $20 股票 sensitivity自動唔同。

**Layer 3 — 4 階段動量分類**（過完 Minervini + VCP 之後，由 EMA20 主導）

| 階段 | 條件 | 動作 |
|---|---|---|
| 🌱 **早期轉強** | EMA20 由走平翹上、EMA5↗EMA20 金叉 | 觀察 |
| 💪 **強勢** | EMA20 斜率 3-7%（45°甜蜜區）、跑贏 SPY | 加 watchlist |
| ⏳ **回調** | 強勢股回到 EMA20 附近、成交縮量 | 等突破 |
| 🎯 **入場** | 陽燭企返 EMA20、量回升、RSI 40-60 | **市價入** |

**Layer 4 — 期權異動 confirmation**

技術面好 + Smart money方向一致 = 最高 conviction。例如：
- 入場 bucket + Minervini 8/8 + VCP Ready + Options flow 聰明錢睇升 → **四層 confirm，A+ setup**
- 入場 bucket + Minervini 8/8 + Options flow 聰明錢睇跌 → **可能有壞消息，skip**


### 離場邏輯（建埋喺 Stock Detail）

- **初始止蝕**：10日swing low再落2%，最闊 -7%
- **跟價止賺**：趨勢未死用 EMA20，跌穿 EMA50 全走
- **全撤訊號**：以下任何 2 個齊出 — 收市跌穿 50日線、RSI<30 反彈無力、放量陰跌、M頂/H&S
"""
)

st.info(
    "📡 **數據源**：Yahoo Finance（免費，無 API key）· "
    "**股池**：可揀 99 / 500 / 517 隻 · "
    "**更新方式**：每日第一次開Screener自動拎當日收市最新數據 · "
    "**成本**：$0"
)

st.warning(
    "⏱️ **首次掃描提示**：呢個 Streamlit Cloud 免費 deploy 嘅 app 有兩個延遲：\n"
    "1. 如果幾個鐘冇人開 → app 會sleep，wake up需要30-60秒\n"
    "2. 首次scan要1-2分鐘（要由Yahoo拎晒當日數據）\n\n"
    "**建議**：第一次開先揀 **「精選動量股（~99隻）」**（sidebar 揀），秒返結果。"
    "想睇全市場再揀 **S&P 500+NDX**。每日後續訪問會用cache，秒速load。"
)

st.markdown("---")
st.caption(
    "⚠️ 研究工具，唔係投資建議。每注 size 按止蝕距離計，唔好賭超過你輸得起。"
)
