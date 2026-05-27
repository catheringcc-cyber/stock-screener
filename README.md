# 動量 + 回調選股 Web App

一個跟住「3階段動量 + 回調」策略嘅美股screener，用Python + Streamlit + Yahoo Finance（**全免費，唔洗API key**）。

## 功能

- **📊 Screener** — 每日掃 S&P 500 + Nasdaq 100（517隻），分4個bucket（入場 / 回調 / 強勢觀察 / 🌱轉強候選）
- **🔍 個股分析** — K線圖配 EMA5/20/50、RSI、EMA20斜率指標、HV波幅百分位、自動計入場/止蝕/目標價、跟價止賺建議、**期權異動分析**
- **📉 回測** — 揀任何一隻股，walk-forward模擬策略表現
- **🎯 期權異動** — 掃~120隻活躍options名嘅近月（60日內到期）期權鏈，揾PCR異常 + 大額$flow + Vol/OI異動，偵測insider-like activity

## 快速開始（本機）

```bash
# 一次過搞掂環境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 跑app
streamlit run app.py
```

開browser去 `http://localhost:8501`。第一次掃要1-2分鐘（下載517隻股票數據），之後讀cache好快。

## 點分享俾人review

### 方法 1：Streamlit Community Cloud（最推薦，免費永久online）

1. 建立GitHub repo，push晒呢個project上去（exclude `.venv` 同 `.cache`）
2. 去 https://share.streamlit.io 用GitHub登入
3. 撳「New app」→ 揀你個repo → main file填 `app.py`
4. 撳Deploy，等3-5分鐘
5. 攞到一個 `xxx.streamlit.app` 嘅public URL，直接send link俾人

✅ 24/7 online · 免費 · 對方click條link就可以用，唔洗install任何嘢

### 方法 2：ngrok tunnel（最快，但你部機要開住）

```bash
# 一次性setup
brew install ngrok
# 去 https://dashboard.ngrok.com/get-started/your-authtoken 攞個免費token
ngrok config add-authtoken <你個token>

# 然後每次想share：先開streamlit，再開ngrok
streamlit run app.py            # terminal 1
ngrok http 8501                 # terminal 2
```

ngrok會印一條 `https://xxxx.ngrok-free.app` 嘅URL，send俾人click就睇到。

⚠️ 只有你部機開住先有效，關咗就斷。

### 方法 3：截圖／錄screencast

最低成本嘅方法 — Mac上：
- 截圖：`Cmd + Shift + 4` 然後拖出範圍
- 錄影：`Cmd + Shift + 5` 揀「Record Selected Portion」

### 方法 4：分享source code

呢個zip已經整好喺 `~/stock-screener.zip`，22KB，可以email、Google Drive、或者上傳GitHub。對方解壓後跟住「快速開始」嗰part跑就得。

## 策略邏輯

**主指標係 EMA20**（reaction快過SMA，適合swing trade）。**EMA20斜率（10日%變化）係單一最高alpha指標** —— 3-7%/10d = 45°甜蜜區。

| 階段 | 條件 | 點解 |
|---|---|---|
| **🌱 早期轉強 Emerging** | EMA20剛由走平翹上、EMA5↗EMA20金叉、成交有量 | 預測未來2-4週會變強勢 |
| **💪 強勢 Strong** | EMA20斜率正常、企EMA20&50之上、3M跑贏SPY、近期放量 | 確認趨勢；等回調 |
| **⏳ 回調 Pullback** | 強勢股回到EMA20附近、成交縮量 | 自然回調；**黃金位 = 斜率高+距離近** |
| **🎯 入場 Entry** | 陽燭企返EMA20、量回升、RSI 40-60 | 回調完真係轉勢 |

每階段攞 0-100 分。Bucket優先級：Entry > Pullback > Strong > Emerging。

### 自動計嘅levels
- **入場**：「入場」bucket用市價；其他bucket用近期20日high做breakout entry
- **止蝕**：10日swing low落2%，封頂 -7%（避免大蝕）
- **目標1**：入場 + 2×風險（2R）
- **目標2**：量度升幅（近期base嘅幅度向上投射）

### 持貨中嘅跟價止賺
- 趨勢未死：用20日線做trailing stop（跌穿減半倉）
- 最後防線：50日線（跌穿全走）

## 期權異動偵測（🎯 期權異動 page）

掃~120隻活躍options名單，揾「市場有冇異常買call/put」嘅pattern：

| 指標 | 解讀 |
|---|---|
| **PCR (Put/Call Volume Ratio)** | >1.5 看淡傾斜 · <0.4 看好傾斜 · 中間正常 |
| **Vol/OI 比率** | 今日成交 / 之前OI，≥2 = 新開大倉，≥5 = 高度可疑 |
| **$ Notional flow** | vol × 價 × 100，揾大資金壓注位置 |

**Filter**：只睇60日內到期 + strike離spot ±10% 至 +40%（避開深ITM box trades同遠OTM lottery）

⚠️ **限制**：
- yfinance對活躍股嘅OI經常返0（data lag），Vol/OI route有時唔fire — 但PCR + $flow已夠sharp
- IV數值可能唔準（特別係流動性差嘅strike），用作參考但唔好做主要判斷
- 第一日run只得當下snapshot，**經過幾日累積先有歷史baseline** 可以計z-score


## 限制（值得留意）

1. 數據係Yahoo Finance EOD（股票）+ near-real-time（期權），**唔係professional grade**
2. Scoring係rule-based heuristic，**唔係ML概率**。60+分當值得睇、70+分當高信心
3. 回測係單股嘅，未做portfolio-level sizing
4. 唔包括股息／earnings／消息面 — 純技術 + flow分析

## 檔案結構

```
stock-screener/
├── app.py                      # Streamlit首頁
├── pages/
│   ├── 1_📊_選股.py           # Screener (4 buckets)
│   ├── 2_🔍_個股分析.py        # Stock detail + 期權異動
│   ├── 3_📉_回測.py           # Backtest
│   └── 4_🎯_期權異動.py        # Options anomaly scanner
├── lib/
│   ├── universe.py            # S&P 500 + NDX股票list
│   ├── data.py                # yfinance下載 + cache
│   ├── indicators.py          # EMA, SMA, RSI, slope, HV percentile
│   ├── strategy.py            # 4階段scoring (emerging/strong/pullback/entry)
│   ├── options.py             # Options chain + PCR + unusual contracts
│   └── backtest.py            # Walk-forward backtester
├── requirements.txt
└── README.md
```

## 免責

研究用途。技術分析永遠唔代表必賺。每注size要按止蝕距離計，永遠唔好賭超過你輸得起嘅錢。
