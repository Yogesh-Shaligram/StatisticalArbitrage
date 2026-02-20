import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import pytz
import warnings
import json
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cloud StatArb Bot", layout="wide", page_icon="☁️")

# ---------------------------------------------------------
# 1. GOOGLE SHEETS CLOUD STORAGE LOGIC
# ---------------------------------------------------------
# Define the Google Sheets Scope and Authenticate via Streamlit Secrets
scopes = ["https://www.googleapis.com/auth/spreadsheets"]


@st.cache_resource
def get_gspread_client():
    # Streamlit Cloud securely injects your Google Credentials from the Secrets manager
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


# Connect to your specific Google Sheet (Replace with your actual exact Sheet name later)
SHEET_NAME = "StatArb_Trading_Database"
client = get_gspread_client()
sheet = client.open(SHEET_NAME)
state_tab = sheet.worksheet("State")
ledger_tab = sheet.worksheet("Ledger")


def load_cloud_state():
    """Pulls your capital and positions down from Google Sheets."""
    try:
        raw_data = state_tab.acell('A1').value
        if raw_data:
            state_data = json.loads(raw_data)
            st.session_state.portfolio = state_data.get('portfolio', 1000000.0)

            # Reconstruct the states dictionary
            loaded_states = {}
            for k, v in state_data.get('states', {}).items():
                a1, a2 = k.split('|')
                loaded_states[(a1, a2)] = v
            st.session_state.states = loaded_states
        else:
            raise ValueError("Empty cell")
    except Exception:
        # If the cell is empty or broken, start fresh
        st.session_state.portfolio = 1000000.0

        # We define pairs later, but we need to initialize them here
        st.session_state.states = {}

        # Load Ledger
    try:
        records = ledger_tab.get_all_records()
        st.session_state.trade_log = records
    except Exception:
        st.session_state.trade_log = []


def save_cloud_state():
    """Pushes your capital and positions up to Google Sheets Cell A1."""
    str_states = {f"{k[0]}|{k[1]}": v for k, v in st.session_state.states.items()}
    state_data = {
        'portfolio': st.session_state.portfolio,
        'states': str_states
    }
    # Dump the entire memory state into a single cell to avoid complex formatting
    state_tab.update_acell('A1', json.dumps(state_data))


def append_to_cloud_ledger(trade_dict):
    """Adds a new row to the Google Sheets Ledger tab."""
    # Convert dictionary values to a simple list for Google Sheets
    row_data = [
        trade_dict['Time'], trade_dict['Pair'], trade_dict['Asset'],
        trade_dict['Action'], trade_dict['Price'], trade_dict['Qty'], trade_dict['P&L']
    ]
    ledger_tab.append_row(row_data)


# ---------------------------------------------------------
# 2. MARKET HOURS GATEKEEPER
# ---------------------------------------------------------
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

is_weekday = now_ist.weekday() < 5
current_time = now_ist.time()
market_open_time = datetime.strptime("09:15", "%H:%M").time()
market_close_time = datetime.strptime("15:30", "%H:%M").time()

MARKET_IS_OPEN = is_weekday and (market_open_time <= current_time <= market_close_time)

# ---------------------------------------------------------
# 3. Watchlist & Initialization
# ---------------------------------------------------------
pairs = [
    ('AXISBANK.NS', 'BANKBARODA.NS'), ('KOTAKBANK.NS', 'BANKBARODA.NS'),
    ('DRREDDY.NS', 'LUPIN.NS'), ('DABUR.NS', 'GODREJCP.NS'),
    ('KOTAKBANK.NS', 'PNB.NS'), ('MARUTI.NS', 'TVSMOTOR.NS'),
    ('JSWSTEEL.NS', 'COALINDIA.NS'), ('HDFCBANK.NS', 'FEDERALBNK.NS'),
    ('M&M.NS', 'EICHERMOT.NS'), ('KOTAKBANK.NS', 'FEDERALBNK.NS'),
    ('BAJFINANCE.NS', 'BAJAJFINSV.NS'), ('TCS.NS', 'PERSISTENT.NS'),
    ('HDFCBANK.NS', 'ICICIBANK.NS'), ('BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'),
    ('RELIANCE.NS', 'TATAPOWER.NS'), ('RELIANCE.NS', 'ONGC.NS')
]
all_tickers = list(set([ticker for pair in pairs for ticker in pair]))

ENTRY_Z = 1.75
EXIT_Z = 0.0
TRADE_ALLOCATION = 50000.0

if 'portfolio' not in st.session_state:
    load_cloud_state()
    # Ensure states are mapped to the pairs list if it was an empty boot
    if not st.session_state.states:
        st.session_state.states = {pair: {'position': 0, 'units': 0, 'entry_price': 0.0} for pair in pairs}


@st.cache_data
def calibrate_pairs():
    hist_data = yf.download(all_tickers, period="1y", progress=False)['Close'].ffill().dropna()
    calibrated = {}
    for a1, a2 in pairs:
        model = sm.OLS(hist_data[a1], hist_data[a2]).fit()
        calibrated[(a1, a2)] = model.params.iloc[0]
    return calibrated


calibrated_pairs = calibrate_pairs()

# ---------------------------------------------------------
# 4. Live Data Fetching & Dashboard UI
# ---------------------------------------------------------
timestamp = now_ist.strftime("%Y-%m-%d %H:%M:%S")
live_data = yf.download(all_tickers, period="40d", progress=False)['Close'].ffill()

with st.sidebar:
    st.header("☁️ Cloud Control Panel")
    if MARKET_IS_OPEN:
        st.success("🟢 MARKET OPEN - Trading Active")
    else:
        st.error("🔴 MARKET CLOSED - Trading Paused")
        st.caption("Engine locked. Data synced to Google Sheets.")
    st.markdown("---")
    st.info("Your trade ledger is safely stored in Google Sheets.")

st.title("☁️ Cloud Statistical Arbitrage Dashboard")
st.caption(f"Last updated: {timestamp} IST | Synced with Google Sheets")

col1, col2, col3 = st.columns(3)
col1.metric("Cloud Capital", f"₹{st.session_state.portfolio:,.2f}")
col2.metric("Active Pairs", sum(1 for state in st.session_state.states.values() if state['position'] != 0))
col3.metric("Total Completed Trades", len([t for t in st.session_state.trade_log if t['Action'] == 'SELL']))

st.markdown("---")

# ---------------------------------------------------------
# 5. Core Trading Logic
# ---------------------------------------------------------
dynamic_titles = []
for a1, a2 in pairs:
    state = st.session_state.states[(a1, a2)]
    base_title = f"{a1.replace('.NS', '')} / {a2.replace('.NS', '')}"
    if state['position'] != 0:
        held_asset = a1.replace('.NS', '') if state['position'] == 1 else a2.replace('.NS', '')
        base_title += f" | [HOLDING: {state['units']} {held_asset}]"
    dynamic_titles.append(base_title)

fig = make_subplots(rows=4, cols=4, subplot_titles=dynamic_titles)
alerts = []
row, col = 1, 1
state_changed = False

for asset1, asset2 in pairs:
    ratio = calibrated_pairs[(asset1, asset2)]
    spread_series = live_data[asset1] - (ratio * live_data[asset2])
    sma_series = spread_series.rolling(window=20).mean()
    std_series = spread_series.rolling(window=20).std().replace(0, np.nan)
    z_score_series = ((spread_series - sma_series) / std_series).dropna()

    if not z_score_series.empty:
        current_z = z_score_series.iloc[-1]
        live_p1 = live_data[asset1].iloc[-1]
        live_p2 = live_data[asset2].iloc[-1]

        pair_state = st.session_state.states[(asset1, asset2)]
        short_name1, short_name2 = asset1.replace('.NS', ''), asset2.replace('.NS', '')

        # --- CLOUD TRADING LOGIC ---
        if MARKET_IS_OPEN:
            if pair_state['position'] == 0:
                if current_z < -ENTRY_Z:
                    units = int(TRADE_ALLOCATION // live_p1)
                    if units > 0 and st.session_state.portfolio >= (units * live_p1):
                        st.session_state.portfolio -= (units * live_p1)
                        st.session_state.states[(asset1, asset2)] = {'position': 1, 'units': units,
                                                                     'entry_price': live_p1}

                        trade_dict = {'Time': timestamp, 'Pair': f"{short_name1}/{short_name2}", 'Asset': short_name1,
                                      'Action': 'BUY', 'Price': round(live_p1, 2), 'Qty': units, 'P&L': 0.0}
                        st.session_state.trade_log.append(trade_dict)
                        append_to_cloud_ledger(trade_dict)  # Push to Sheets
                        alerts.append(f"🚨 BOUGHT {units} {short_name1}")
                        state_changed = True

                elif current_z > ENTRY_Z:
                    units = int(TRADE_ALLOCATION // live_p2)
                    if units > 0 and st.session_state.portfolio >= (units * live_p2):
                        st.session_state.portfolio -= (units * live_p2)
                        st.session_state.states[(asset1, asset2)] = {'position': 2, 'units': units,
                                                                     'entry_price': live_p2}

                        trade_dict = {'Time': timestamp, 'Pair': f"{short_name1}/{short_name2}", 'Asset': short_name2,
                                      'Action': 'BUY', 'Price': round(live_p2, 2), 'Qty': units, 'P&L': 0.0}
                        st.session_state.trade_log.append(trade_dict)
                        append_to_cloud_ledger(trade_dict)  # Push to Sheets
                        alerts.append(f"🚨 BOUGHT {units} {short_name2}")
                        state_changed = True

            elif pair_state['position'] == 1:
                if current_z > EXIT_Z:
                    revenue = pair_state['units'] * live_p1
                    profit = revenue - (pair_state['units'] * pair_state['entry_price'])
                    st.session_state.portfolio += revenue

                    trade_dict = {'Time': timestamp, 'Pair': f"{short_name1}/{short_name2}", 'Asset': short_name1,
                                  'Action': 'SELL', 'Price': round(live_p1, 2), 'Qty': pair_state['units'],
                                  'P&L': round(profit, 2)}
                    st.session_state.trade_log.append(trade_dict)
                    append_to_cloud_ledger(trade_dict)  # Push to Sheets

                    st.session_state.states[(asset1, asset2)] = {'position': 0, 'units': 0, 'entry_price': 0.0}
                    alerts.append(f"🔔 SOLD {short_name1} | Profit: ₹{profit:.2f}")
                    state_changed = True

            elif pair_state['position'] == 2:
                if current_z < EXIT_Z:
                    revenue = pair_state['units'] * live_p2
                    profit = revenue - (pair_state['units'] * pair_state['entry_price'])
                    st.session_state.portfolio += revenue

                    trade_dict = {'Time': timestamp, 'Pair': f"{short_name1}/{short_name2}", 'Asset': short_name2,
                                  'Action': 'SELL', 'Price': round(live_p2, 2), 'Qty': pair_state['units'],
                                  'P&L': round(profit, 2)}
                    st.session_state.trade_log.append(trade_dict)
                    append_to_cloud_ledger(trade_dict)  # Push to Sheets

                    st.session_state.states[(asset1, asset2)] = {'position': 0, 'units': 0, 'entry_price': 0.0}
                    alerts.append(f"🔔 SOLD {short_name2} | Profit: ₹{profit:.2f}")
                    state_changed = True

        # --- PLOTTING ---
        plot_data = z_score_series.tail(20)
        line_color = 'rgba(0, 200, 0, 1)' if st.session_state.states[(asset1, asset2)][
                                                 'position'] != 0 else 'rgba(0, 0, 255, 0.7)'

        fig.add_trace(
            go.Scatter(x=plot_data.index, y=plot_data.values, mode='lines', line=dict(color=line_color, width=2),
                       showlegend=False), row=row, col=col)
        fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red", line_width=1, row=row, col=col)
        fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red", line_width=1, row=row, col=col)
        fig.add_hline(y=EXIT_Z, line_dash="dot", line_color="gray", line_width=1, row=row, col=col)
        fig.update_yaxes(range=[-3.5, 3.5], row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)

    col += 1
    if col > 4:
        col = 1
        row += 1

# If memory changed, push the new portfolio state to Google Sheets immediately
if state_changed:
    save_cloud_state()

# Apply Market Closed Watermark (Visible right now because it's past 3:30 PM IST)
if not MARKET_IS_OPEN:
    fig.add_annotation(
        text="MARKET CLOSED<br>TRADING PAUSED",
        xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(size=60, color="rgba(255, 0, 0, 0.15)", family="Arial Black"),
        align="center", textangle=-15, showarrow=False
    )

fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(240,240,240,0.5)')
st.plotly_chart(fig, use_container_width=True)

if alerts:
    for alert in alerts:
        if "BOUGHT" in alert:
            st.warning(alert)
        else:
            st.success(alert)

time.sleep(60)
st.rerun()