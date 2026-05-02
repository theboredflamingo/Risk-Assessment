"""
BankQuant · ML Dashboard  (light mode)
Run with: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from river import tree, metrics, drift
from datetime import datetime
import json, os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankQuant · ML Dashboard",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — warm light theme ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #f7f4ef;
    --surface:   #ffffff;
    --surface2:  #f0ece4;
    --border:    #e2ddd5;
    --border2:   #ccc8be;
    --ink:       #1a1714;
    --ink2:      #3d3a35;
    --muted:     #8c8880;
    --accent:    #1a56a0;
    --accent2:   #e8f0fb;
    --green:     #0d7a45;
    --green-bg:  #e8f5ee;
    --red:       #c0392b;
    --red-bg:    #fdf0ee;
    --amber:     #b45309;
    --amber-bg:  #fef3e2;
    --rule:      #d4cfc7;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.6rem 2.2rem 2.4rem; max-width: 1640px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 0.73rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.55rem 1rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88; }

/* Masthead */
.masthead {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.7rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: baseline;
    gap: 1.2rem;
}
.masthead-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1;
}
.masthead-sub {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-left: 1px solid var(--rule);
    padding-left: 1.2rem;
}
.masthead-date {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
}

/* KPI strip */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
.kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.15rem 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1;
    color: var(--ink);
}
.kpi-value.up   { color: var(--green); }
.kpi-value.down { color: var(--red); }
.kpi-value.acc  { color: var(--accent); }
.kpi-value.warn { color: var(--amber); }
.kpi-pill {
    display: inline-block;
    margin-top: 0.45rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 0.18rem 0.55rem;
    border-radius: 99px;
}
.pill-up   { background: var(--green-bg); color: var(--green); }
.pill-down { background: var(--red-bg);   color: var(--red);   }
.pill-neu  { background: var(--accent2);  color: var(--accent); }
.pill-warn { background: var(--amber-bg); color: var(--amber);  }

/* Section titles */
.sec-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.75rem;
    margin-top: 0.2rem;
}

/* Log box */
.log-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink2);
    max-height: 215px;
    overflow-y: auto;
    line-height: 1.75;
}
.log-box span.ts  { color: var(--muted); }
.log-box span.ev  { color: var(--accent); font-weight: 600; }
.log-box span.ok  { color: var(--green); font-weight: 600; }
.log-box span.bad { color: var(--red);   font-weight: 600; }

/* Footer */
.footer {
    margin-top: 1.8rem;
    border-top: 1px solid var(--rule);
    padding-top: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.68rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STOCKS = {
    "HDFCBANK.NS": 0,
    "ICICIBANK.NS": 1,
    "SBIN.NS":      2,
    "AXISBANK.NS":  3,
    "KOTAKBANK.NS": 4,
}
TICKER_LABELS = {
    "HDFCBANK.NS":  "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS":      "SBI",
    "AXISBANK.NS":  "Axis Bank",
    "KOTAKBANK.NS": "Kotak Bank",
}
LOG_FILE = "run_log.json"

# Light-mode Plotly base
PL = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f7f4ef",
    font=dict(family="Inter", color="#1a1714", size=11.5),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#e2ddd5", zeroline=False, linecolor="#d4cfc7", showline=True),
    yaxis=dict(gridcolor="#e2ddd5", zeroline=False, linecolor="#d4cfc7", showline=True),
)

COLORS = ["#1a56a0", "#0d7a45", "#b45309", "#c0392b", "#6b3fa0"]

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, interval="1d", period=period,
                     auto_adjust=True, progress=False)
    df = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.squeeze().dropna()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_niftybank(period="1y"):
    nifty = yf.download("^NSEBANK", interval="1d", period=period,
                        auto_adjust=True, progress=False)
    nifty = nifty[['Close']].apply(pd.to_numeric, errors='coerce')
    nifty.columns = ['Close_nifty']
    return nifty.squeeze().dropna()

def compute_features(df, nifty_df, stock_id):
    df = df.join(nifty_df, how='inner').copy()
    df['log_return']     = np.log(df['Close'] / df['Close'].shift(1))
    df['ret_21d']        = np.log(df['Close'] / df['Close'].shift(21))
    df['nifty_ret_21d']  = np.log(df['Close_nifty'] / df['Close_nifty'].shift(21))
    df['rel_strength']   = df['ret_21d'] - df['nifty_ret_21d']
    ema9  = df['Close'].ewm(span=9,  adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_ratio']      = ema9 / ema50
    delta = df['Close'].diff()
    rs    = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-9)
    df['rsi']            = (100 - 100/(1+rs)).clip(30, 70)
    df['vol_z']          = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)
    df['volatility_20d'] = df['log_return'].rolling(20).std()
    df['stock_id']       = stock_id
    return df.dropna()

def signal_A(prob_up, threshold=0.3):
    return 0 if prob_up < threshold else None

def run_online_learning(df_all, threshold=0.3):

    import pickle
    model = pickle.load(open("dailymfinal.pkl", "rb"))
    detector = drift.ADWIN()
    acc_m    = metrics.Accuracy()
    skip     = {'log_return','target','Open','High','Low','Close','Volume','Close_nifty'}
    feature_cols = [c for c in df_all.columns if c not in skip]
    df_all['target'] = (df_all['log_return'] > 0).astype(int)

    rows = []
    trades = wins = 0
    pnl = 0.0

    for idx, row in df_all.iterrows():
        x       = {c: row[c] for c in feature_cols}
        target  = int(row['target'])
        proba   = model.predict_proba_one(x)
        prob_up = proba.get(1, 0.5)
        signal  = signal_A(prob_up, threshold)

        trade_flag = drift_flag = 0
        trade_pnl  = 0.0

        if signal is not None:
            trades    += 1
            trade_flag = 1
            trade_pnl  = 1.0 if target == 0 else -1.0
            pnl       += trade_pnl
            if trade_pnl > 0:
                wins += 1

        pred = 1 if prob_up > 0.5 else 0
        acc_m.update(target, pred)
        detector.update(int(pred != target))
        if detector.drift_detected:
            drift_flag = 1

        model.learn_one(x, target)

        rows.append({
            "date": idx,
            "stock_id": int(row['stock_id']),
            "ticker": [t for t,s in STOCKS.items() if s==int(row['stock_id'])][0],
            "prob_up": prob_up, "pred": pred, "target": target,
            "accuracy": acc_m.get(), "drift": drift_flag,
            "trade": trade_flag, "trade_pnl": trade_pnl,
            "cum_pnl": pnl, "cum_trades": trades,
            "win_rate": wins/trades if trades>0 else 0,
            "rsi": row['rsi'], "ema_ratio": row['ema_ratio'],
            "rel_strength": row['rel_strength'],
            "volatility_20d": row['volatility_20d'],
            "vol_z": row['vol_z'], "close": row['Close'],
        })

    with open("dailymfinal.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model checkpoint.")
    

    return pd.DataFrame(rows), {
        "accuracy": acc_m.get(), "trades": trades,
        "win_rate": wins/trades if trades>0 else 0, "pnl": pnl,
    }


def load_log():
    runs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                try: runs.append(json.loads(line.strip()))
                except: pass
    return runs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Libre Baskerville",serif;font-size:1.1rem;
                font-weight:700;color:#1a1714;margin-bottom:0.1rem'>₹ BankQuant</div>
    <div style='font-size:0.7rem;color:#8c8880;margin-bottom:1.5rem;
                text-transform:uppercase;letter-spacing:0.1em'>ML Trading Dashboard</div>
    """, unsafe_allow_html=True)

    period         = st.selectbox("Data Period", ["6mo","1y","2y"], index=1)
    focus_ticker   = st.selectbox("Focus Stock", list(STOCKS.keys()),
                                  format_func=lambda t: TICKER_LABELS[t])
    sell_threshold = st.slider("Sell Signal Threshold (prob_up <)", 0.10, 0.50, 0.30, 0.05)

    st.markdown("---")
    st.button("▶  Run / Refresh Pipeline", use_container_width=True)
    st.markdown("<div style='font-size:0.68rem;color:#8c8880;margin-top:0.5rem;line-height:1.5'>Downloads live data, replays the online learning loop, and recomputes all metrics.</div>",
                unsafe_allow_html=True)

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='masthead'>
  <div class='masthead-title'>₹ BankQuant · Online Learning Monitor</div>
  <div class='masthead-sub'>Hoeffding Adaptive Tree · ADWIN Drift · NSE Banking Universe</div>
  <div class='masthead-date'>{datetime.now().strftime('%d %b %Y  %H:%M')}</div>
</div>
""", unsafe_allow_html=True)

# ── Build data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def build_all(period):
    nifty = fetch_niftybank(period)
    frames = []
    for ticker, sid in STOCKS.items():
        df = fetch_data(ticker, period)
        frames.append(compute_features(df, nifty, sid))
    return pd.concat(frames).sort_index()

with st.spinner("Fetching market data & replaying online-learning loop …"):
    raw_df = build_all(period)
    results, summary = run_online_learning(raw_df.copy(), sell_threshold)

results['signal'] = results['prob_up'].apply(lambda p: 0 if p < sell_threshold else None)
drift_events = results[results['drift'] == 1]

# ── KPI strip ─────────────────────────────────────────────────────────────────
acc_pct  = summary['accuracy'] * 100
wr_pct   = summary['win_rate'] * 100
pnl_val  = summary['pnl']
n_drift  = int(results['drift'].sum())
n_trades = summary['trades']

acc_cls = "up"   if acc_pct  >= 55 else "warn" if acc_pct  >= 50 else "down"
wr_cls  = "up"   if wr_pct   >= 50 else "down"
pnl_cls = "up"   if pnl_val  > 0   else "down"
dr_cls  = "warn" if n_drift  > 0   else "up"

st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi'>
    <div class='kpi-label'>Model Accuracy</div>
    <div class='kpi-value {acc_cls}'>{acc_pct:.1f}%</div>
    <span class='kpi-pill {"pill-up" if acc_cls=="up" else "pill-warn" if acc_cls=="warn" else "pill-down"}'>
      {"Above" if acc_pct>=50 else "Below"} baseline
    </span>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Sell Signals</div>
    <div class='kpi-value acc'>{n_trades}</div>
    <span class='kpi-pill pill-neu'>prob_up &lt; {sell_threshold}</span>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Win Rate</div>
    <div class='kpi-value {wr_cls}'>{wr_pct:.1f}%</div>
    <span class='kpi-pill {"pill-up" if wr_pct>=50 else "pill-down"}'>On sell signals</span>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Cum. PnL (pts)</div>
    <div class='kpi-value {pnl_cls}'>{int(pnl_val):+d}</div>
    <span class='kpi-pill {"pill-up" if pnl_val>0 else "pill-down"}'>+1 / −1 scoring</span>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Drift Events</div>
    <div class='kpi-value {dr_cls}'>{n_drift}</div>
    <span class='kpi-pill {"pill-warn" if n_drift>0 else "pill-up"}'>ADWIN detections</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Row 1 — Candlestick + RSI  |  Accuracy + PnL ─────────────────────────────
col1, col2 = st.columns([3, 2], gap="medium")

with col1:
    st.markdown("<div class='sec-title'>Price Chart · EMA 9/50 · Sell Signals · Drift Alerts</div>", unsafe_allow_html=True)
    focus     = results[results['ticker'] == focus_ticker].copy()
    raw_focus = raw_df[raw_df['stock_id'] == STOCKS[focus_ticker]].copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=raw_focus.index,
        open=raw_focus['Open'], high=raw_focus['High'],
        low=raw_focus['Low'],   close=raw_focus['Close'],
        increasing_line_color='#0d7a45', decreasing_line_color='#c0392b',
        increasing_fillcolor='#0d7a45', decreasing_fillcolor='#c0392b',
        line_width=1, name="Price"
    ), row=1, col=1)

    ema9  = raw_focus['Close'].ewm(span=9,  adjust=False).mean()
    ema50 = raw_focus['Close'].ewm(span=50, adjust=False).mean()
    fig.add_trace(go.Scatter(x=raw_focus.index, y=ema9,
        line=dict(color='#1a56a0', width=1.4), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_focus.index, y=ema50,
        line=dict(color='#b45309', width=1.4, dash='dot'), name="EMA 50"), row=1, col=1)

    sell_f = focus[focus['signal'] == 0]
    if not sell_f.empty:
        close_at_sell = raw_focus['Close'].reindex(sell_f['date']).values
        fig.add_trace(go.Scatter(
            x=sell_f['date'], y=close_at_sell, mode='markers',
            marker=dict(symbol='triangle-down', size=10,
                        color='#c0392b', line=dict(width=1, color='#fff')),
            name="Sell Signal"
        ), row=1, col=1)

    drift_f = focus[focus['drift'] == 1]
    if not drift_f.empty:
        close_at_drift = raw_focus['Close'].reindex(drift_f['date']).values
        fig.add_trace(go.Scatter(
            x=drift_f['date'], y=close_at_drift, mode='markers',
            marker=dict(symbol='star', size=12,
                        color='#b45309', line=dict(width=1, color='#fff')),
            name="Drift ⚠"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=focus['date'], y=focus['rsi'],
        line=dict(color='#6b3fa0', width=1.5), name="RSI(14)"
    ), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#ccc8be", row=2, col=1)

    fig.update_layout(**PL, height=460, showlegend=True,
                      legend=dict(orientation="h", y=1.05, x=0,
                                  bgcolor="rgba(255,255,255,0.85)",
                                  bordercolor="#e2ddd5", borderwidth=1,
                                  font=dict(size=11)),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="RSI (30–70)", row=2, col=1, title_font_size=10)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    st.markdown("<div class='sec-title'>Rolling Online Accuracy · All Stocks</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    for i, (ticker, sid) in enumerate(STOCKS.items()):
        sub = results[results['ticker'] == ticker]
        fig2.add_trace(go.Scatter(
            x=sub['date'], y=sub['accuracy']*100,
            name=TICKER_LABELS[ticker],
            line=dict(color=COLORS[i], width=1.6)
        ))
    fig2.add_hline(y=50, line_dash="dash", line_color="#ccc8be",
                   annotation_text="50% baseline",
                   annotation_font=dict(color="#8c8880", size=10))
    fig2.update_layout(**PL, height=215,
                       legend=dict(orientation="h", y=1.1, font=dict(size=10),
                                   bgcolor="rgba(255,255,255,0)"),
                       yaxis_title="Accuracy (%)")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='sec-title' style='margin-top:0.5rem'>Cumulative PnL · All Stocks</div>", unsafe_allow_html=True)
    cum = results.groupby('date')['trade_pnl'].sum().cumsum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=cum['date'], y=cum['trade_pnl'],
        fill='tozeroy', mode='lines',
        line=dict(color='#0d7a45', width=2),
        fillcolor='rgba(13,122,69,0.10)', name="PnL"
    ))
    fig3.add_hline(y=0, line_color="#ccc8be", line_width=1)
    fig3.update_layout(**PL, height=195, showlegend=False, yaxis_title="Points")
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ── Row 2 — prob_up hist | Heatmap | Vol-Z ───────────────────────────────────
col3, col4, col5 = st.columns([1.3, 2, 1.3], gap="medium")

with col3:
    st.markdown("<div class='sec-title'>prob_up Distribution</div>", unsafe_allow_html=True)
    focus_prob = results[results['ticker'] == focus_ticker]
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=focus_prob['prob_up'], nbinsx=28,
        marker=dict(color='#1a56a0', opacity=0.65,
                    line=dict(color='#f7f4ef', width=0.6)),
        name="prob_up"
    ))
    fig4.add_vline(x=sell_threshold, line_dash="dash", line_color="#c0392b", line_width=1.5,
                   annotation_text=f"Sell < {sell_threshold}",
                   annotation_font=dict(color="#c0392b", size=10))
    fig4.add_vline(x=0.5, line_dash="dot", line_color="#8c8880", line_width=1)
    fig4.update_layout(**PL, height=240, showlegend=False,
                       xaxis_title="prob_up", yaxis_title="Count")
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

with col4:
    st.markdown("<div class='sec-title'>Feature Correlation Heatmap · Last 200 Obs.</div>", unsafe_allow_html=True)
    feat_cols = ['rsi','ema_ratio','rel_strength','volatility_20d','vol_z','prob_up']
    corr_data = results[feat_cols].dropna().tail(200).corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns.tolist(),
        y=corr_data.index.tolist(),
        colorscale=[[0,'#c0392b'],[0.5,'#f7f4ef'],[1,'#0d7a45']],
        zmin=-1, zmax=1,
        text=np.round(corr_data.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10.5, color='#1a1714'),
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.9, tickfont=dict(size=9))
    ))
    fig5.update_layout(**PL, height=240)
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

with col5:
    st.markdown("<div class='sec-title'>Volume Z-Score · Last 60 Days</div>", unsafe_allow_html=True)
    foc    = results[results['ticker'] == focus_ticker].tail(60)
    vz_col = ['#0d7a45' if v > 0 else '#c0392b' for v in foc['vol_z']]
    fig6   = go.Figure(go.Bar(
        x=foc['date'], y=foc['vol_z'],
        marker_color=vz_col, marker_line_width=0,
    ))
    fig6.add_hline(y=0, line_color="#ccc8be", line_width=1)
    fig6.update_layout(**PL, height=240, showlegend=False, yaxis_title="Z-Score")
    st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

# ── Row 3 — Relative strength | Drift timeline ───────────────────────────────
col6, col7 = st.columns(2, gap="medium")

with col6:
    st.markdown("<div class='sec-title'>Relative Strength vs Nifty Bank · 21-Day Window</div>", unsafe_allow_html=True)
    fig7 = go.Figure()
    for i, (ticker, sid) in enumerate(STOCKS.items()):
        sub = results[results['ticker'] == ticker]
        fig7.add_trace(go.Scatter(
            x=sub['date'], y=sub['rel_strength'],
            name=TICKER_LABELS[ticker],
            line=dict(color=COLORS[i], width=1.6)
        ))
    fig7.add_hline(y=0, line_dash="dot", line_color="#ccc8be")
    fig7.update_layout(**PL, height=240,
                       legend=dict(orientation="h", y=1.12, font=dict(size=10),
                                   bgcolor="rgba(255,255,255,0)"),
                       yaxis_title="Log-return excess")
    st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

with col7:
    st.markdown("<div class='sec-title'>Drift Events Timeline · ADWIN Detections</div>", unsafe_allow_html=True)
    if not drift_events.empty:
        drift_pivot = drift_events.groupby(['date','ticker'])['drift'].sum().reset_index()
        fig8 = go.Figure()
        for i, (ticker, sid) in enumerate(STOCKS.items()):
            sub = drift_pivot[drift_pivot['ticker'] == ticker]
            if not sub.empty:
                fig8.add_trace(go.Scatter(
                    x=sub['date'],
                    y=[TICKER_LABELS[ticker]] * len(sub),
                    mode='markers',
                    marker=dict(symbol='diamond', size=11,
                                color=COLORS[i],
                                line=dict(width=1, color='#fff')),
                    name=TICKER_LABELS[ticker]
                ))
        fig8.update_layout(**PL, height=240, showlegend=False, xaxis_title="Date")
    else:
        fig8 = go.Figure()
        fig8.add_annotation(text="No drift detected in this period",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(color="#8c8880", size=13))
        fig8.update_layout(**PL, height=240)
    st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})

# ── Row 4 — Per-stock table | Run log ────────────────────────────────────────
col8, col9 = st.columns([2, 1.5], gap="medium")

with col8:
    st.markdown("<div class='sec-title'>Per-Stock Model Summary</div>", unsafe_allow_html=True)
    rows_table = []
    for ticker, sid in STOCKS.items():
        sub      = results[results['ticker'] == ticker]
        sell_sub = sub[sub['signal'] == 0]
        wins_sub = sell_sub[sell_sub['trade_pnl'] > 0]
        rows_table.append({
            "Stock":       TICKER_LABELS[ticker],
            "Accuracy %":  f"{sub['accuracy'].iloc[-1]*100:.1f}",
            "Signals":     len(sell_sub),
            "Win Rate %":  f"{(len(wins_sub)/max(len(sell_sub),1))*100:.1f}",
            "Net PnL":     f"{sell_sub['trade_pnl'].sum():+.0f}",
            "Drift Hits":  int(sub['drift'].sum()),
            "Avg prob_up": f"{sub['prob_up'].mean():.3f}",
            "Avg RSI":     f"{sub['rsi'].mean():.1f}",
        })
    summary_df = pd.DataFrame(rows_table).set_index("Stock")

    def style_pnl(v):
        try:
            n = float(v)
            return 'color: #0d7a45; font-weight:600' if n > 0 else 'color: #c0392b; font-weight:600' if n < 0 else ''
        except: return ''

    styled = (summary_df.style
        .applymap(style_pnl, subset=["Net PnL"])
        .set_properties(**{
            'background-color': '#ffffff',
            'color': '#1a1714',
            'border-color': '#e2ddd5',
            'font-size': '13px',
            'font-family': 'Inter, sans-serif',
        })
        .set_table_styles([{
            'selector': 'th',
            'props': [
                ('background-color','#f7f4ef'),
                ('color','#8c8880'),
                ('font-size','11px'),
                ('text-transform','uppercase'),
                ('letter-spacing','0.07em'),
                ('font-weight','600'),
                ('border-bottom','2px solid #e2ddd5'),
            ]
        }])
    )
    st.dataframe(styled, use_container_width=True)

with col9:
    st.markdown("<div class='sec-title'>Historical Run Log</div>", unsafe_allow_html=True)
    log_runs = load_log()
    if log_runs:
        log_df = pd.DataFrame(log_runs)
        log_df['time'] = pd.to_datetime(log_df['time'])
        log_df = log_df.sort_values('time', ascending=False).head(10)
        log_html = "<div class='log-box'>"
        for _, r in log_df.iterrows():
            acc_c = "ok"  if r['accuracy'] >= 0.5 else "bad"
            pnl_c = "ok"  if r['pnl'] > 0         else "bad"
            log_html += (
                f"<div><span class='ts'>{r['time'].strftime('%m-%d %H:%M')}</span>  "
                f"<span class='ev'>acc</span> <span class='{acc_c}'>{r['accuracy']:.3f}</span>  "
                f"<span class='ev'>pnl</span> <span class='{pnl_c}'>{int(r['pnl']):+d}</span>  "
                f"<span class='ev'>wr</span> {r['win_rate']:.2f}  "
                f"<span class='ev'>tr</span> {int(r['trades'])}</div>"
            )
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='log-box' style='color:#8c8880'>No run_log.json found.<br>Run the pipeline script first.</div>",
                    unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='footer'>
  <span>BankQuant · Hoeffding Adaptive Tree · ADWIN Drift · NSE Banking Universe</span>
  <span>Last refresh · {datetime.now().strftime('%d %b %Y  %H:%M')}</span>
</div>
""", unsafe_allow_html=True)