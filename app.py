import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

# ─── Page Config ────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="StockSenseAI | Stock Predictor",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

from PIL import Image

# logo = Image.open(r"D:\Umair Data\Build Projects\StockSenseAI\logo pic\logo.png")

logo = Image.open("logo pic/logo.png")

st.set_page_config(
    page_title="StockSenseAI | Stock Predictor",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar toggle button visible
st.markdown("""
<style>
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
button[kind="header"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    background: #0d1117 !important;
    border: 1.5px solid #00e5a0 !important;
    border-radius: 8px !important;
}
button[kind="header"] svg {
    fill: #00e5a0 !important;
    color: #00e5a0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════
   FORCED DARK MODE — overrides Windows/browser theme
   ══════════════════════════════════════════════════ */
html { color-scheme: dark !important; }

.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main,
.main > div,
.stMainBlockContainer,
[data-testid="stMain"] {
    background-color: #080c10 !important;
    color: #cdd9e5 !important;
}

/* Force text color */
p, label, li,
[class*="css"], [class*="st-"] {
    color: #cdd9e5 !important;
}

/* Dropdown popup background */
ul[data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #111820 !important;
    color: #cdd9e5 !important;
}

/* ── Root tokens ── */
:root {
    --bg:        #080c10;
    --bg2:       #0d1117;
    --bg3:       #111820;
    --border:    #1e2d3d;
    --green:     #00e5a0;
    --green-dim: #00a872;
    --red:       #ff4d6d;
    --red-dim:   #c03050;
    --gold:      #f5c842;
    --text:      #cdd9e5;
    --text-dim:  #637380;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
    --radius:    10px;
    --glow:      0 0 20px rgba(0,229,160,0.15);
}

/* ── Reset & base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
}


/* Collapse button container */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display:    flex !important;
    visibility: visible !important;
    opacity:    1 !important;
    position:   relative !important;
    z-index:    999999 !important;
}

/* Button base */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    display:         flex !important;
    align-items:     center !important;
    justify-content: center !important;
    visibility:      visible !important;
    opacity:         1 !important;
    background:      #0d1117 !important;
    border:          1.5px solid #00e5a0 !important;
    border-radius:   8px !important;
    width:           2.2rem !important;
    height:          2.2rem !important;
    cursor:          pointer !important;
    box-shadow:      0 0 12px rgba(0,229,160,0.2) !important;
    color:           #00e5a0 !important;
    font-size:       1.1rem !important;
    position:        relative !important;
    overflow:        visible !important;
}

/* ✅ Hide ALL original content inside button (broken SVG/spans) */
[data-testid="stSidebarCollapseButton"] button *,
[data-testid="stSidebarCollapsedControl"] button * {
    display: none !important;
}

/* ✅ Inject a visible arrow icon via pseudo-element */
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after {
    content:     "‹" !important;   /* left arrow for collapse */
    display:     flex !important;
    align-items: center !important;
    justify-content: center !important;
    color:       #00e5a0 !important;
    font-size:   1.4rem !important;
    font-weight: bold !important;
    line-height: 1 !important;
}

/* Hover */
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
    background:  rgba(0,229,160,0.12) !important;
    box-shadow:  0 0 20px rgba(0,229,160,0.35) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div {
    background:       #161b22 !important;
    background-color: #161b22 !important;
    border-right:     1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family:    var(--sans) !important;
    font-size:      0.7rem !important;
    font-weight:    700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color:          var(--text-dim) !important;
    border-bottom:  1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom:  1rem !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background:       var(--bg3) !important;
    background-color: var(--bg3) !important;
    border:           1px solid var(--border) !important;
    border-radius:    var(--radius) !important;
    transition:       border-color 0.2s, box-shadow 0.2s !important;
}
div[data-baseweb="select"]:hover {
    border-color: var(--green) !important;
    box-shadow:   var(--glow) !important;
}
div[data-baseweb="select"] * {
    font-family:      var(--mono) !important;
    color:            var(--text) !important;
    background:       transparent !important;
    background-color: transparent !important;
}

/* ── Primary predict button ── */
.stButton > button {
    width:            100% !important;
    background:       transparent !important;
    background-color: transparent !important;
    border:           1.5px solid var(--green) !important;
    color:            var(--green) !important;
    font-family:      var(--mono) !important;
    font-size:        0.85rem !important;
    font-weight:      700 !important;
    letter-spacing:   0.1em !important;
    text-transform:   uppercase !important;
    padding:          0.75rem 1.5rem !important;
    border-radius:    var(--radius) !important;
    cursor:           pointer !important;
    transition:       background 0.2s, box-shadow 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background:  rgba(0,229,160,0.08) !important;
    box-shadow:  0 0 28px rgba(0,229,160,0.25) !important;
    transform:   translateY(-1px) !important;
}
.stButton > button:active {
    transform:  translateY(0px) !important;
    box-shadow: 0 0 10px rgba(0,229,160,0.15) !important;
}

/* ── Metric cards ── */
[data-testid="stMetricValue"] {
    font-family:    var(--mono) !important;
    font-size:      1.8rem !important;
    color:          var(--green) !important;
}
[data-testid="stMetricLabel"] {
    font-family:    var(--sans) !important;
    font-size:      0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color:          var(--text-dim) !important;
}
[data-testid="stMetric"] {
    background:       var(--bg3) !important;
    background-color: var(--bg3) !important;
    border:           1px solid var(--border) !important;
    border-radius:    var(--radius) !important;
    padding:          1rem 1.25rem !important;
}

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
    border-radius:    var(--radius) !important;
    font-family:      var(--sans) !important;
    border:           1px solid !important;
    background-color: var(--bg2) !important;
}
div[data-testid="stAlert"][class*="info"] {
    background:   rgba(0,229,160,0.05) !important;
    border-color: var(--green-dim) !important;
    color:        var(--text) !important;
}
div[data-testid="stAlert"][class*="success"] {
    background:   rgba(0,229,160,0.07) !important;
    border-color: var(--green) !important;
}
div[data-testid="stAlert"][class*="error"] {
    background:   rgba(255,77,109,0.07) !important;
    border-color: var(--red) !important;
}
div[data-testid="stAlert"][class*="warning"] {
    background:   rgba(245,200,66,0.06) !important;
    border-color: var(--gold) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border:           1px solid var(--border) !important;
    border-radius:    var(--radius) !important;
    overflow:         hidden !important;
    background-color: var(--bg2) !important;
}
.stDataFrame * { font-family: var(--mono) !important; font-size: 0.78rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green) !important; }

/* ── Headings ── */
h1, h2, h3 {
    font-family:    var(--sans) !important;
    font-weight:    800 !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.3rem !important; color: var(--text) !important; }
h3 { font-size: 1rem !important;   color: var(--text-dim) !important; }

/* ── Line chart ── */
[data-testid="stArrowVegaLiteChart"] canvas { border-radius: var(--radius); }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--green-dim), var(--green)) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00e5a0; }
</style>
""", unsafe_allow_html=True)

# ─── Stock metadata ─────────────────────────────────────────────────────────
STOCKS = {
    "AAPL":  {"emoji": "🍎", "name": "Apple Inc.",              "sector": "Technology",  "desc": "Consumer electronics & software giant.",       "logo": "https://www.google.com/s2/favicons?domain=apple.com&sz=64"},
    "MSFT":  {"emoji": "🪟", "name": "Microsoft Corp.",         "sector": "Technology",  "desc": "Cloud, enterprise software & AI leader.",      "logo": "https://www.google.com/s2/favicons?domain=microsoft.com&sz=64"},
    "GOOGL": {"emoji": "🔍", "name": "Alphabet (Google)",       "sector": "Technology",  "desc": "Search, advertising & cloud services.",        "logo": "https://www.google.com/s2/favicons?domain=google.com&sz=64"},
    "TSLA":  {"emoji": "⚡", "name": "Tesla Inc.",              "sector": "EV / Energy", "desc": "Electric vehicles & clean energy.",            "logo": "https://www.google.com/s2/favicons?domain=tesla.com&sz=64"},
    "JPM":   {"emoji": "🏦", "name": "JPMorgan Chase",          "sector": "Financials",  "desc": "Largest US bank by assets.",                   "logo": "https://www.google.com/s2/favicons?domain=jpmorganchase.com&sz=64"},
    "JNJ":   {"emoji": "💊", "name": "Johnson & Johnson",       "sector": "Healthcare",  "desc": "Pharmaceuticals & medical devices.",           "logo": "https://www.google.com/s2/favicons?domain=jnj.com&sz=64"},
}

FEATURES = [
    'Return', 'Volatility', 'Momentum',
    'MA_10_slope', 'MA_50_slope',
    'Price_vs_MA10', 'Price_vs_MA50',
    'Volume_change', 'High_Low_range',
]
import base64

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("""
    # <div style='text-align:center; padding: 1rem 0 1.5rem;'>
    #     <div style='font-size:2.4rem;'>📈</div>
    #     <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.3rem; color:#cdd9e5; letter-spacing:-0.02em;'>StockSenseAI</div>
    #     <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#637380; letter-spacing:0.15em; text-transform:uppercase; margin-top:2px;'>ML Stock Predictor</div>
    # </div>
    # """, unsafe_allow_html=True)

    # logo_base64 = base64.b64encode(open(r"D:\Umair Data\Build Projects\StockSenseAI\logo pic\logo 2.png", "rb").read()).decode()
    # Try this exact name (with space)
    # logo_base64 = base64.b64encode(open(r"D:\Umair Data\Build Projects\StockSenseAI\logo pic\logo 2.png", "rb").read()).decode()
    # st.markdown(f"""
    # <div style='text-align:center; padding: 1rem 0 1.5rem;'>
    #     <img src='data:image/png;base64,{logo_base64}' 
    #          style='height:80px; width:80px; border-radius:50%; object-fit:cover;'/>
    #     <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.3rem; color:#cdd9e5; letter-spacing:-0.02em;'>StockSenseAI</div>
    #     <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#637380; letter-spacing:0.15em; text-transform:uppercase; margin-top:2px;'>ML Stock Predictor</div>
    # </div>
    # """, unsafe_allow_html=True)
    
    logo_base64 = base64.b64encode(
    open("logo pic/logo 2.png", "rb").read()
).decode()

    st.markdown(f"""
<div style='text-align:center; padding: 1rem 0 1.5rem;'>
    <img src='data:image/png;base64,{logo_base64}' 
         style='height:80px; width:80px; border-radius:50%; object-fit:cover;'/>
    <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.3rem; color:#cdd9e5; letter-spacing:-0.02em;'>StockSenseAI</div>
    <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#637380; letter-spacing:0.15em; text-transform:uppercase; margin-top:2px;'>ML Stock Predictor</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("## How It Works")
    st.markdown("""
    <div style='font-size:0.82rem; color:#9aa8b6; line-height:1.7; font-family:"Syne",sans-serif;'>
    <p>The model analyzes <strong style='color:#cdd9e5;'>9 engineered features</strong> derived from 6 months of OHLCV data:</p>
    <ul style='padding-left:1.1rem; margin:0.5rem 0;'>
        <li><strong style='color:#00e5a0;'>Return</strong> – Daily % price change</li>
        <li><strong style='color:#00e5a0;'>Volatility</strong> – 10-day rolling std dev</li>
        <li><strong style='color:#00e5a0;'>Momentum</strong> – Price vs MA10</li>
        <li><strong style='color:#00e5a0;'>MA Slopes</strong> – Trend direction of MAs</li>
        <li><strong style='color:#00e5a0;'>Price vs MAs</strong> – Relative position</li>
        <li><strong style='color:#00e5a0;'>Volume Change</strong> – Buying pressure</li>
        <li><strong style='color:#00e5a0;'>High-Low Range</strong> – Intraday volatility</li>
    </ul>
    <p>A <strong style='color:#cdd9e5;'>Random Forest classifier</strong> trained on AAPL, MSFT, GOOGL, TSLA, JPM, JNJ outputs a binary <em>UP / DOWN</em> signal for the next trading day.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Supported Tickers")
    for sym, meta in STOCKS.items():
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; padding:7px 10px; "
            f"background:#111820; border-radius:8px; margin-bottom:6px; border:1px solid #1e2d3d;'>"
            f"<img src='{meta['logo']}' width='20' height='20' "
            f"style='border-radius:4px; background:#fff; padding:2px; object-fit:contain;' "
            f"onerror=\"this.style.display='none'\"/>"
            f"<span style='color:#00e5a0; font-weight:700; font-family:\"Space Mono\",monospace; font-size:0.78rem;'>{sym}</span>"
            f"<span style='color:#637380; font-size:0.72rem;'>– {meta['sector']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#3d5066; font-family:\"Space Mono\",monospace; line-height:1.6;'>"
        "⚠️ For educational use only.<br>Not financial advice.<br>Past signals ≠ future results."
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Main Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.25rem;'>
    <span style='font-family:"Space Mono",monospace; font-size:0.7rem; color:#00e5a0; letter-spacing:0.2em; text-transform:uppercase;'>
        ML-POWERED · NEXT-DAY SIGNAL
    </span>
</div>
""", unsafe_allow_html=True)
st.title("Stock Direction Predictor")
st.markdown(
    "<p style='font-size:1rem; color:#637380; margin-top:-0.5rem; font-family:\"Syne\",sans-serif;'>"
    "Select a stock to see live feature analysis and a next-day UP / DOWN prediction."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ─── Stock selector ──────────────────────────────────────────────────────────
col_sel, col_info = st.columns([1, 1.8], gap="large")

with col_sel:
    st.markdown("#### 01 — Choose Stock")
    ticker = st.selectbox(
        "Ticker",
        options=list(STOCKS.keys()),
        format_func=lambda x: f"{STOCKS[x]['emoji']}  {x} — {STOCKS[x]['name']}",
        label_visibility="collapsed",
    )

info = STOCKS[ticker]

with col_info:
    st.markdown("#### Stock Profile")
    st.markdown(
        f"""
        <div style='background:#0d1117; border:1px solid #1e2d3d; border-radius:10px; padding:1rem 1.25rem;'>
            <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;'>
                <img src='{info['logo']}' width='36' height='36'
                     style='border-radius:6px; background:#fff; padding:3px; object-fit:contain;'
                     onerror="this.style.display='none'"/>
                <div>
                    <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.05rem; color:#cdd9e5;'>{info['name']}</div>
                    <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#00e5a0; letter-spacing:0.1em;'>{ticker} · {info['sector']}</div>
                </div>
            </div>
            <p style='font-size:0.82rem; color:#9aa8b6; margin:0; line-height:1.6; font-family:"Syne",sans-serif;'>{info['desc']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ─── Pre-prediction explainer ────────────────────────────────────────────────
st.markdown("#### 02 — What Will Be Predicted?")

exp_cols = st.columns(3, gap="medium")
explainers = [
    ("🎯", "Binary Signal", "The model outputs either <strong style='color:#00e5a0;'>UP ↑</strong> or <strong style='color:#ff4d6d;'>DOWN ↓</strong> — whether tomorrow's closing price will be higher or lower than today's."),
    ("📊", "Confidence Score", "Alongside the signal, you get the model's <strong style='color:#f5c842;'>probability score</strong> (0–100%). Higher confidence = stronger signal conviction."),
    ("🧮", "Feature Snapshot", "View the <strong style='color:#cdd9e5;'>live computed features</strong> for the latest trading day — the exact inputs the model uses to make its decision."),
]
for col, (icon, title, desc) in zip(exp_cols, explainers):
    with col:
        st.markdown(
            f"""
            <div style='background:#0d1117; border:1px solid #1e2d3d; border-radius:10px; padding:1rem 1.1rem; height:100%;'>
                <div style='font-size:1.4rem; margin-bottom:0.4rem;'>{icon}</div>
                <div style='font-family:"Syne",sans-serif; font-weight:700; font-size:0.88rem; color:#cdd9e5; margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.78rem; color:#9aa8b6; line-height:1.6; font-family:"Syne",sans-serif;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Predict button ──────────────────────────────────────────────────────────
st.markdown("#### 03 — Run Prediction")

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button(f"⚡  Predict {ticker} Tomorrow's Direction")

# ─── Prediction logic ─────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner(f"Fetching {ticker} data & computing features…"):
        try:
            model = joblib.load("models/stock_model_1.pkl")

            df = yf.download(ticker, period="6mo", progress=False)
            df.columns = df.columns.droplevel(1)

            df['Return']         = df['Close'].pct_change()
            df['MA_10']          = df['Close'].rolling(10).mean()
            df['MA_50']          = df['Close'].rolling(50).mean()
            df['Volatility']     = df['Return'].rolling(10).std()
            df['Momentum']       = df['Close'] - df['MA_10']
            df['MA_10_slope']    = df['MA_10'].pct_change()
            df['MA_50_slope']    = df['MA_50'].pct_change()
            df['Price_vs_MA10']  = df['Close'] / df['MA_10']
            df['Price_vs_MA50']  = df['Close'] / df['MA_50']
            df['Volume_change']  = df['Volume'].pct_change()
            df['High_Low_range'] = (df['High'] - df['Low']) / df['Close']
            df.dropna(inplace=True)

            latest        = df[FEATURES].iloc[-1:]
            prediction    = model.predict(latest)[0]
            probability   = model.predict_proba(latest)[0]
            confidence    = max(probability) * 100
            prob_up       = probability[1] * 100
            prob_down     = probability[0] * 100
            current_price = float(df['Close'].iloc[-1])

            st.divider()

            # ── Result banner ──
            if prediction == 1:
                st.markdown(
                    f"""
                    <div style='background:rgba(0,229,160,0.06); border:1.5px solid #00e5a0;
                         border-radius:12px; padding:1.5rem 1.75rem; margin-bottom:1rem;
                         box-shadow:0 0 32px rgba(0,229,160,0.12);'>
                        <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
                             letter-spacing:0.2em; text-transform:uppercase; color:#00a872; margin-bottom:0.4rem;'>
                            TOMORROW'S SIGNAL
                        </div>
                        <div style='display:flex; align-items:center; gap:12px;'>
                            <img src='{info['logo']}' width='32' height='32'
                                 style='border-radius:6px; background:#fff; padding:3px; object-fit:contain;'
                                 onerror="this.style.display='none'"/>
                            <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:2rem;
                                 color:#00e5a0; letter-spacing:-0.02em;'>
                                📈 BULLISH — Price likely UP
                            </div>
                        </div>
                        <div style='font-family:"Space Mono",monospace; font-size:0.8rem; color:#9aa8b6; margin-top:0.4rem;'>
                            Model confidence: <strong style='color:#00e5a0;'>{confidence:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='background:rgba(255,77,109,0.06); border:1.5px solid #ff4d6d;
                         border-radius:12px; padding:1.5rem 1.75rem; margin-bottom:1rem;
                         box-shadow:0 0 32px rgba(255,77,109,0.12);'>
                        <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
                             letter-spacing:0.2em; text-transform:uppercase; color:#c03050; margin-bottom:0.4rem;'>
                            TOMORROW'S SIGNAL
                        </div>
                        <div style='display:flex; align-items:center; gap:12px;'>
                            <img src='{info['logo']}' width='32' height='32'
                                 style='border-radius:6px; background:#fff; padding:3px; object-fit:contain;'
                                 onerror="this.style.display='none'"/>
                            <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:2rem;
                                 color:#ff4d6d; letter-spacing:-0.02em;'>
                                📉 BEARISH — Price likely DOWN
                            </div>
                        </div>
                        <div style='font-family:"Space Mono",monospace; font-size:0.8rem; color:#9aa8b6; margin-top:0.4rem;'>
                            Model confidence: <strong style='color:#ff4d6d;'>{confidence:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Metrics row ──
            m1, m2, m3, m4 = st.columns(4, gap="medium")
            with m1:
                st.metric("Current Price", f"${current_price:.2f}")
            with m2:
                st.metric("UP Probability", f"{prob_up:.1f}%")
            with m3:
                st.metric("DOWN Probability", f"{prob_down:.1f}%")
            with m4:
                st.metric("Confidence", f"{confidence:.1f}%")

            # ── Probability bar ──
            st.markdown("<br>**Signal Probability Breakdown**", unsafe_allow_html=True)
            st.progress(int(prob_up))
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; font-family:\"Space Mono\",monospace; font-size:0.72rem; margin-top:-0.3rem;'>"
                f"<span style='color:#637380;'>0%</span>"
                f"<span style='color:#00e5a0;'>UP {prob_up:.1f}%</span>"
                f"<span style='color:#ff4d6d;'>DOWN {prob_down:.1f}%</span>"
                f"<span style='color:#637380;'>100%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Charts ──
            chart_col, feat_col = st.columns([1.5, 1], gap="large")

            with chart_col:
                st.markdown("#### 📉 Price Chart — Last 6 Months")
                st.line_chart(df['Close'], height=260, use_container_width=True)

            with feat_col:
                st.markdown("#### 🧮 Feature Snapshot (Latest Day)")
                feat_df = latest.T.rename(columns={latest.index[-1]: "Value"})
                feat_df["Value"] = feat_df["Value"].round(5)
                st.dataframe(feat_df, use_container_width=True, height=260)

            st.divider()

            # ── Recent data ──
            st.markdown("#### 📊 Recent OHLCV + Indicators (Last 10 Days)")
            display_df = df[['Close', 'Volume', 'Return', 'Volatility', 'Momentum']].tail(10).round(4)
            st.dataframe(display_df, use_container_width=True)

            # ── Disclaimer ──
            st.warning(
                "⚠️ **Disclaimer:** QuantEdge is a machine-learning demo built for educational purposes. "
                "Predictions are probabilistic and based on historical patterns. "
                "This is **not** financial advice. Always do your own research before investing."
            )

        except FileNotFoundError:
            st.error(
                "❌ **Model file not found.** Make sure `stock_model_1.pkl` is in the same directory as this script."
            )
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")






# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import joblib

# # ─── Page Config ────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="QuantEdge | Stock Predictor",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─── Global CSS ─────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

# /* ── Root tokens ── */
# :root {
#     --bg:        #080c10;
#     --bg2:       #0d1117;
#     --bg3:       #111820;
#     --border:    #1e2d3d;
#     --green:     #00e5a0;
#     --green-dim: #00a872;
#     --red:       #ff4d6d;
#     --red-dim:   #c03050;
#     --gold:      #f5c842;
#     --text:      #cdd9e5;
#     --text-dim:  #637380;
#     --mono:      'Space Mono', monospace;
#     --sans:      'Syne', sans-serif;
#     --radius:    10px;
#     --glow:      0 0 20px rgba(0,229,160,0.15);
# }

# /* ── Reset & base ── */
# html, body, [class*="css"] {
#     background-color: var(--bg) !important;
#     color: var(--text) !important;
#     font-family: var(--sans) !important;
# }

# /* ── Hide Streamlit chrome ── */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1100px !important; }

# /* ── Sidebar ── */
# section[data-testid="stSidebar"] {
#     background: var(--bg2) !important;
#     border-right: 1px solid var(--border) !important;
# }
# section[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
# section[data-testid="stSidebar"] .stMarkdown h2 {
#     font-family: var(--sans) !important;
#     font-size: 0.7rem !important;
#     font-weight: 700 !important;
#     letter-spacing: 0.15em !important;
#     text-transform: uppercase !important;
#     color: var(--text-dim) !important;
#     border-bottom: 1px solid var(--border);
#     padding-bottom: 0.5rem;
#     margin-bottom: 1rem !important;
# }

# /* ── Selectbox ── */
# div[data-baseweb="select"] {
#     background: var(--bg3) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: var(--radius) !important;
#     transition: border-color 0.2s, box-shadow 0.2s !important;
# }
# div[data-baseweb="select"]:hover {
#     border-color: var(--green) !important;
#     box-shadow: var(--glow) !important;
# }
# div[data-baseweb="select"] * {
#     font-family: var(--mono) !important;
#     color: var(--text) !important;
#     background: transparent !important;
# }

# /* ── Primary predict button ── */
# .stButton > button {
#     width: 100% !important;
#     background: transparent !important;
#     border: 1.5px solid var(--green) !important;
#     color: var(--green) !important;
#     font-family: var(--mono) !important;
#     font-size: 0.85rem !important;
#     font-weight: 700 !important;
#     letter-spacing: 0.1em !important;
#     text-transform: uppercase !important;
#     padding: 0.75rem 1.5rem !important;
#     border-radius: var(--radius) !important;
#     cursor: pointer !important;
#     transition: background 0.2s, box-shadow 0.2s, transform 0.1s !important;
#     position: relative !important;
#     overflow: hidden !important;
# }
# .stButton > button:hover {
#     background: rgba(0,229,160,0.08) !important;
#     box-shadow: 0 0 28px rgba(0,229,160,0.25) !important;
#     transform: translateY(-1px) !important;
# }
# .stButton > button:active {
#     transform: translateY(0px) !important;
#     box-shadow: 0 0 10px rgba(0,229,160,0.15) !important;
# }

# /* ── Metric cards ── */
# [data-testid="stMetricValue"] {
#     font-family: var(--mono) !important;
#     font-size: 1.8rem !important;
#     color: var(--green) !important;
# }
# [data-testid="stMetricLabel"] {
#     font-family: var(--sans) !important;
#     font-size: 0.75rem !important;
#     letter-spacing: 0.1em !important;
#     text-transform: uppercase !important;
#     color: var(--text-dim) !important;
# }
# [data-testid="stMetric"] {
#     background: var(--bg3) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: var(--radius) !important;
#     padding: 1rem 1.25rem !important;
# }

# /* ── Alert boxes ── */
# div[data-testid="stAlert"] {
#     border-radius: var(--radius) !important;
#     font-family: var(--sans) !important;
#     border: 1px solid !important;
# }

# /* ── Info ── */
# div[data-testid="stAlert"][class*="info"] {
#     background: rgba(0,229,160,0.05) !important;
#     border-color: var(--green-dim) !important;
#     color: var(--text) !important;
# }

# /* ── Success ── */
# div[data-testid="stAlert"][class*="success"] {
#     background: rgba(0,229,160,0.07) !important;
#     border-color: var(--green) !important;
# }

# /* ── Error ── */
# div[data-testid="stAlert"][class*="error"] {
#     background: rgba(255,77,109,0.07) !important;
#     border-color: var(--red) !important;
# }

# /* ── Warning ── */
# div[data-testid="stAlert"][class*="warning"] {
#     background: rgba(245,200,66,0.06) !important;
#     border-color: var(--gold) !important;
# }

# /* ── Dataframe ── */
# .stDataFrame {
#     border: 1px solid var(--border) !important;
#     border-radius: var(--radius) !important;
#     overflow: hidden !important;
# }
# .stDataFrame * { font-family: var(--mono) !important; font-size: 0.78rem !important; }

# /* ── Divider ── */
# hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

# /* ── Spinner ── */
# .stSpinner > div { border-top-color: var(--green) !important; }

# /* ── Headings ── */
# h1, h2, h3 {
#     font-family: var(--sans) !important;
#     font-weight: 800 !important;
#     letter-spacing: -0.02em !important;
# }
# h1 { font-size: 2.2rem !important; }
# h2 { font-size: 1.3rem !important; color: var(--text) !important; }
# h3 { font-size: 1rem !important; color: var(--text-dim) !important; }

# /* ── Line chart ── */
# [data-testid="stArrowVegaLiteChart"] canvas { border-radius: var(--radius); }

# /* ── Progress bar ── */
# .stProgress > div > div > div > div {
#     background: linear-gradient(90deg, var(--green-dim), var(--green)) !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─── Stock metadata ─────────────────────────────────────────────────────────
# STOCKS = {
#     "AAPL":  {"emoji": "🍎", "name": "Apple Inc.",              "sector": "Technology",    "desc": "Consumer electronics & software giant."},
#     "MSFT":  {"emoji": "🪟", "name": "Microsoft Corp.",         "sector": "Technology",    "desc": "Cloud, enterprise software & AI leader."},
#     "GOOGL": {"emoji": "🔍", "name": "Alphabet (Google)",       "sector": "Technology",    "desc": "Search, advertising & cloud services."},
#     "TSLA":  {"emoji": "⚡", "name": "Tesla Inc.",              "sector": "EV / Energy",   "desc": "Electric vehicles & clean energy."},
#     "JPM":   {"emoji": "🏦", "name": "JPMorgan Chase",          "sector": "Financials",    "desc": "Largest US bank by assets."},
#     "JNJ":   {"emoji": "💊", "name": "Johnson & Johnson",       "sector": "Healthcare",    "desc": "Pharmaceuticals & medical devices."},
# }

# FEATURES = [
#     'Return', 'Volatility', 'Momentum',
#     'MA_10_slope', 'MA_50_slope',
#     'Price_vs_MA10', 'Price_vs_MA50',
#     'Volume_change', 'High_Low_range',
# ]

# # ─── Sidebar ────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style='text-align:center; padding: 1rem 0 1.5rem;'>
#         <div style='font-size:2.4rem;'>📈</div>
#         <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.3rem; color:#cdd9e5; letter-spacing:-0.02em;'>QuantEdge</div>
#         <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#637380; letter-spacing:0.15em; text-transform:uppercase; margin-top:2px;'>ML Stock Predictor</div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("## How It Works")
#     st.markdown("""
#     <div style='font-size:0.82rem; color:#9aa8b6; line-height:1.7; font-family:"Syne",sans-serif;'>
#     <p>The model analyzes <strong style='color:#cdd9e5;'>9 engineered features</strong> derived from 6 months of OHLCV data:</p>
#     <ul style='padding-left:1.1rem; margin:0.5rem 0;'>
#         <li><strong style='color:#00e5a0;'>Return</strong> – Daily % price change</li>
#         <li><strong style='color:#00e5a0;'>Volatility</strong> – 10-day rolling std dev</li>
#         <li><strong style='color:#00e5a0;'>Momentum</strong> – Price vs MA10</li>
#         <li><strong style='color:#00e5a0;'>MA Slopes</strong> – Trend direction of MAs</li>
#         <li><strong style='color:#00e5a0;'>Price vs MAs</strong> – Relative position</li>
#         <li><strong style='color:#00e5a0;'>Volume Change</strong> – Buying pressure</li>
#         <li><strong style='color:#00e5a0;'>High-Low Range</strong> – Intraday volatility</li>
#     </ul>
#     <p>A <strong style='color:#cdd9e5;'>Random Forest classifier</strong> trained on AAPL, MSFT, GOOGL, TSLA, JPM, JNJ outputs a binary <em>UP / DOWN</em> signal for the next trading day.</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")

#     st.markdown("## Supported Tickers")
#     for ticker, info in STOCKS.items():
#         st.markdown(
#             f"<div style='display:flex; align-items:center; gap:0.5rem; padding:0.3rem 0; font-size:0.8rem; font-family:\"Space Mono\",monospace;'>"
#             f"<span>{info['emoji']}</span>"
#             f"<span style='color:#00e5a0; font-weight:700;'>{ticker}</span>"
#             f"<span style='color:#637380;'>– {info['sector']}</span>"
#             f"</div>",
#             unsafe_allow_html=True,
#         )

#     st.markdown("---")
#     st.markdown(
#         "<div style='font-size:0.7rem; color:#3d5066; font-family:\"Space Mono\",monospace; line-height:1.6;'>"
#         "⚠️ For educational use only.<br>Not financial advice.<br>Past signals ≠ future results."
#         "</div>",
#         unsafe_allow_html=True,
#     )

# # ─── Main Header ────────────────────────────────────────────────────────────
# st.markdown("""
# <div style='margin-bottom:0.25rem;'>
#     <span style='font-family:"Space Mono",monospace; font-size:0.7rem; color:#00e5a0; letter-spacing:0.2em; text-transform:uppercase;'>
#         ML-POWERED · NEXT-DAY SIGNAL
#     </span>
# </div>
# """, unsafe_allow_html=True)
# st.title("Stock Direction Predictor")
# st.markdown(
#     "<p style='font-size:1rem; color:#637380; margin-top:-0.5rem; font-family:\"Syne\",sans-serif;'>"
#     "Select a stock to see live feature analysis and a next-day UP / DOWN prediction."
#     "</p>",
#     unsafe_allow_html=True,
# )

# st.divider()

# # ─── Stock selector ──────────────────────────────────────────────────────────
# col_sel, col_info = st.columns([1, 1.8], gap="large")

# with col_sel:
#     st.markdown("#### 01 — Choose Stock")
#     ticker = st.selectbox(
#         "Ticker",
#         options=list(STOCKS.keys()),
#         format_func=lambda x: f"{STOCKS[x]['emoji']}  {x} — {STOCKS[x]['name']}",
#         label_visibility="collapsed",
#     )

# info = STOCKS[ticker]

# with col_info:
#     st.markdown("#### Stock Profile")
#     st.markdown(
#         f"""
#         <div style='background:#0d1117; border:1px solid #1e2d3d; border-radius:10px; padding:1rem 1.25rem;'>
#             <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;'>
#                 <span style='font-size:1.8rem;'>{info['emoji']}</span>
#                 <div>
#                     <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.05rem; color:#cdd9e5;'>{info['name']}</div>
#                     <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#00e5a0; letter-spacing:0.1em;'>{ticker} · {info['sector']}</div>
#                 </div>
#             </div>
#             <p style='font-size:0.82rem; color:#9aa8b6; margin:0; line-height:1.6; font-family:"Syne",sans-serif;'>{info['desc']}</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

# st.divider()

# # ─── Pre-prediction explainer ────────────────────────────────────────────────
# st.markdown("#### 02 — What Will Be Predicted?")

# exp_cols = st.columns(3, gap="medium")
# explainers = [
#     ("🎯", "Binary Signal", "The model outputs either <strong style='color:#00e5a0;'>UP ↑</strong> or <strong style='color:#ff4d6d;'>DOWN ↓</strong> — whether tomorrow's closing price will be higher or lower than today's."),
#     ("📊", "Confidence Score", "Alongside the signal, you get the model's <strong style='color:#f5c842;'>probability score</strong> (0–100%). Higher confidence = stronger signal conviction."),
#     ("🧮", "Feature Snapshot", "View the <strong style='color:#cdd9e5;'>live computed features</strong> for the latest trading day — the exact inputs the model uses to make its decision."),
# ]
# for col, (icon, title, desc) in zip(exp_cols, explainers):
#     with col:
#         st.markdown(
#             f"""
#             <div style='background:#0d1117; border:1px solid #1e2d3d; border-radius:10px; padding:1rem 1.1rem; height:100%;'>
#                 <div style='font-size:1.4rem; margin-bottom:0.4rem;'>{icon}</div>
#                 <div style='font-family:"Syne",sans-serif; font-weight:700; font-size:0.88rem; color:#cdd9e5; margin-bottom:0.4rem;'>{title}</div>
#                 <div style='font-size:0.78rem; color:#9aa8b6; line-height:1.6; font-family:"Syne",sans-serif;'>{desc}</div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

# st.markdown("<br>", unsafe_allow_html=True)

# # ─── Predict button ──────────────────────────────────────────────────────────
# st.markdown("#### 03 — Run Prediction")

# _, btn_col, _ = st.columns([1, 2, 1])
# with btn_col:
#     predict_clicked = st.button(f"⚡  Predict {ticker} Tomorrow's Direction")

# # ─── Prediction logic ─────────────────────────────────────────────────────────
# if predict_clicked:
#     with st.spinner(f"Fetching {ticker} data & computing features…"):
#         try:
#             model = joblib.load("models/stock_model_1.pkl")

#             df = yf.download(ticker, period="6mo", progress=False)
#             df.columns = df.columns.droplevel(1)

#             df['Return']         = df['Close'].pct_change()
#             df['MA_10']          = df['Close'].rolling(10).mean()
#             df['MA_50']          = df['Close'].rolling(50).mean()
#             df['Volatility']     = df['Return'].rolling(10).std()
#             df['Momentum']       = df['Close'] - df['MA_10']
#             df['MA_10_slope']    = df['MA_10'].pct_change()
#             df['MA_50_slope']    = df['MA_50'].pct_change()
#             df['Price_vs_MA10']  = df['Close'] / df['MA_10']
#             df['Price_vs_MA50']  = df['Close'] / df['MA_50']
#             df['Volume_change']  = df['Volume'].pct_change()
#             df['High_Low_range'] = (df['High'] - df['Low']) / df['Close']
#             df.dropna(inplace=True)

#             latest       = df[FEATURES].iloc[-1:]
#             prediction   = model.predict(latest)[0]
#             probability  = model.predict_proba(latest)[0]
#             confidence   = max(probability) * 100
#             prob_up      = probability[1] * 100
#             prob_down    = probability[0] * 100
#             current_price = float(df['Close'].iloc[-1])

#             st.divider()

#             # ── Result banner ──
#             if prediction == 1:
#                 st.markdown(
#                     f"""
#                     <div style='background:rgba(0,229,160,0.06); border:1.5px solid #00e5a0;
#                          border-radius:12px; padding:1.5rem 1.75rem; margin-bottom:1rem;
#                          box-shadow:0 0 32px rgba(0,229,160,0.12);'>
#                         <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
#                              letter-spacing:0.2em; text-transform:uppercase; color:#00a872; margin-bottom:0.4rem;'>
#                             TOMORROW'S SIGNAL
#                         </div>
#                         <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:2rem;
#                              color:#00e5a0; letter-spacing:-0.02em;'>
#                             📈 BULLISH — Price likely UP
#                         </div>
#                         <div style='font-family:"Space Mono",monospace; font-size:0.8rem; color:#9aa8b6; margin-top:0.4rem;'>
#                             Model confidence: <strong style='color:#00e5a0;'>{confidence:.1f}%</strong>
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 st.markdown(
#                     f"""
#                     <div style='background:rgba(255,77,109,0.06); border:1.5px solid #ff4d6d;
#                          border-radius:12px; padding:1.5rem 1.75rem; margin-bottom:1rem;
#                          box-shadow:0 0 32px rgba(255,77,109,0.12);'>
#                         <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
#                              letter-spacing:0.2em; text-transform:uppercase; color:#c03050; margin-bottom:0.4rem;'>
#                             TOMORROW'S SIGNAL
#                         </div>
#                         <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:2rem;
#                              color:#ff4d6d; letter-spacing:-0.02em;'>
#                             📉 BEARISH — Price likely DOWN
#                         </div>
#                         <div style='font-family:"Space Mono",monospace; font-size:0.8rem; color:#9aa8b6; margin-top:0.4rem;'>
#                             Model confidence: <strong style='color:#ff4d6d;'>{confidence:.1f}%</strong>
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#             # ── Metrics row ──
#             m1, m2, m3, m4 = st.columns(4, gap="medium")
#             with m1:
#                 st.metric("Current Price", f"${current_price:.2f}")
#             with m2:
#                 st.metric("UP Probability", f"{prob_up:.1f}%")
#             with m3:
#                 st.metric("DOWN Probability", f"{prob_down:.1f}%")
#             with m4:
#                 st.metric("Confidence", f"{confidence:.1f}%")

#             # ── Probability bar ──
#             st.markdown("<br>**Signal Probability Breakdown**", unsafe_allow_html=True)
#             st.progress(int(prob_up))
#             st.markdown(
#                 f"<div style='display:flex; justify-content:space-between; font-family:\"Space Mono\",monospace; font-size:0.72rem; margin-top:-0.3rem;'>"
#                 f"<span style='color:#637380;'>0%</span>"
#                 f"<span style='color:#00e5a0;'>UP {prob_up:.1f}%</span>"
#                 f"<span style='color:#ff4d6d;'>DOWN {prob_down:.1f}%</span>"
#                 f"<span style='color:#637380;'>100%</span>"
#                 f"</div>",
#                 unsafe_allow_html=True,
#             )

#             st.divider()

#             # ── Charts ──
#             chart_col, feat_col = st.columns([1.5, 1], gap="large")

#             with chart_col:
#                 st.markdown("#### 📉 Price Chart — Last 6 Months")
#                 st.line_chart(df['Close'], height=260, use_container_width=True)

#             with feat_col:
#                 st.markdown("#### 🧮 Feature Snapshot (Latest Day)")
#                 feat_df = latest.T.rename(columns={latest.index[-1]: "Value"})
#                 feat_df["Value"] = feat_df["Value"].round(5)
#                 st.dataframe(feat_df, use_container_width=True, height=260)

#             st.divider()

#             # ── Recent data ──
#             st.markdown("#### 📊 Recent OHLCV + Indicators (Last 10 Days)")
#             display_df = df[['Close', 'Volume', 'Return', 'Volatility', 'Momentum']].tail(10).round(4)
#             st.dataframe(display_df, use_container_width=True)

#             # ── Disclaimer ──
#             st.warning(
#                 "⚠️ **Disclaimer:** QuantEdge is a machine-learning demo built for educational purposes. "
#                 "Predictions are probabilistic and based on historical patterns. "
#                 "This is **not** financial advice. Always do your own research before investing."
#             )

#         except FileNotFoundError:
#             st.error(
#                 "❌ **Model file not found.** Make sure `stock_model_1.pkl` is in the same directory as this script."
#             )
#         except Exception as e:
#             st.error(f"❌ Something went wrong: {e}")