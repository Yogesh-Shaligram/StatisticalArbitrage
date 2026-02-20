import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import itertools
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PREPARATION: Define Sectors & Bulk Download Data
# ---------------------------------------------------------
sectors = {
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS',
           'LTTS.NS', 'MPHASIS.NS', 'KPITTECH.NS'],

    'Banks': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'PNB.NS',
              'BANKBARODA.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'AUBANK.NS', 'BANDHANBNK.NS'],

    'Fin_Services': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS', 'SHRIRAMFIN.NS', 'MUTHOOTFIN.NS', 'RECLTD.NS',
                     'PFC.NS', 'IREDA.NS', 'JIOFIN.NS', 'MCX.NS'],

    'Auto': ['M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS',
             'ASHOKLEY.NS', 'BOSCHLTD.NS', 'MRF.NS'],

    'Pharma_Health': ['SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS',
                      'TORNTPHARM.NS', 'ZYDUSLIFE.NS', 'MAXHEALTH.NS', 'APOLLOHOSP.NS', 'SYNGENE.NS'],

    'FMCG_Retail': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS', 'DABUR.NS',
                    'GODREJCP.NS', 'MARICO.NS', 'COLPAL.NS', 'VBL.NS', 'TRENT.NS', 'DMART.NS'],

    'Metals_Mining': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS', 'VEDL.NS', 'NMDC.NS',
                      'JINDALSTEL.NS', 'SAIL.NS'],

    'Energy_Oil': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'ADANIGREEN.NS',
                   'ADANIPOWER.NS', 'GAIL.NS', 'IOC.NS', 'BPCL.NS'],

    'Cement_Infra': ['ULTRACEMCO.NS', 'GRASIM.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'DALBHARAT.NS', 'ACC.NS', 'LT.NS'],

    'Chemicals': ['PIDILITIND.NS', 'SRF.NS', 'AARTIIND.NS', 'TATACHEM.NS', 'DEEPAKNTR.NS', 'UPL.NS']
}

pairs = []
all_tickers = set()

for sector, stocks in sectors.items():
    sector_pairs = list(itertools.combinations(stocks, 2))
    pairs.extend(sector_pairs)
    all_tickers.update(stocks)

print(f"Generated {len(pairs)} pairs. Downloading latest 2-year data...")
end_date = datetime.today()
start_date = end_date - timedelta(days=730)

data = yf.download(list(all_tickers), start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'),
                   progress=False)['Close']
data = data.ffill().dropna()

# ---------------------------------------------------------
# PHASE 1: Scan for Valid (Cointegrated) Pairs
# ---------------------------------------------------------
print("\n--- PHASE 1: Scanning for Cointegration ---")
valid_pairs = []

for asset1, asset2 in pairs:
    if asset1 in data.columns and asset2 in data.columns:
        pair_data = data[[asset1, asset2]].dropna()
        if len(pair_data) > 100:
            score, p_value, _ = ts.coint(pair_data[asset1], pair_data[asset2])
            if p_value < 0.05:
                valid_pairs.append((asset1, asset2, p_value))

print(f"✅ Found {len(valid_pairs)} valid mean-reverting pairs.")

# ---------------------------------------------------------
# PHASE 2: Process Long-Only Strategy & Log Trades
# ---------------------------------------------------------
print("\n--- PHASE 2: Processing Z-Score Strategy ---")
results = []
all_trades = []  # Master ledger for the CSV export
ENTRY_Z = 1.75
EXIT_Z = 0.0

for asset1, asset2, p_val in valid_pairs:
    pair_name = f"{asset1.replace('.NS', '')} / {asset2.replace('.NS', '')}"
    pair_data = data[[asset1, asset2]].dropna().copy()

    # 1. Calculate Dynamic Hedge Ratio & Spread
    model = sm.OLS(pair_data[asset1], pair_data[asset2]).fit()
    ratio = model.params.iloc[0]

    pair_data['Spread'] = pair_data[asset1] - (ratio * pair_data[asset2])
    pair_data['SMA'] = pair_data['Spread'].rolling(window=20).mean()
    pair_data['StdDev'] = pair_data['Spread'].rolling(window=20).std()
    pair_data['Z_Score'] = (pair_data['Spread'] - pair_data['SMA']) / pair_data['StdDev']

    pair_data = pair_data.dropna()

    # 2. Long-Only Execution Logic (₹100,000 Capital)
    position = 0
    units = 0.0
    capital = 100000.0
    equity_curve = [{'Date': pair_data.index[0], 'Capital': capital}]

    for index, row in pair_data.iterrows():
        z_score = row['Z_Score']
        p1 = row[asset1]
        p2 = row[asset2]

        # Entry Signals
        if position == 0:
            if z_score < -ENTRY_Z:
                position = 1
                units = capital / p1
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset_Traded': asset1, 'Action': 'BUY', 'Price': round(p1, 2),
                     'Quantity': round(units, 4), 'Trade_Value': round(capital, 2)})
            elif z_score > ENTRY_Z:
                position = 2
                units = capital / p2
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset_Traded': asset2, 'Action': 'BUY', 'Price': round(p2, 2),
                     'Quantity': round(units, 4), 'Trade_Value': round(capital, 2)})

        # Exit Signals
        elif position == 1:
            if z_score > EXIT_Z:
                capital = units * p1
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset_Traded': asset1, 'Action': 'SELL', 'Price': round(p1, 2),
                     'Quantity': round(units, 4), 'Trade_Value': round(capital, 2)})
                equity_curve.append({'Date': index, 'Capital': capital})
                position = 0

        elif position == 2:
            if z_score < EXIT_Z:
                capital = units * p2
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset_Traded': asset2, 'Action': 'SELL', 'Price': round(p2, 2),
                     'Quantity': round(units, 4), 'Trade_Value': round(capital, 2)})
                equity_curve.append({'Date': index, 'Capital': capital})
                position = 0

                # 3. Calculate Performance Metrics
    trades_made = (len(all_trades) // 2)  # Divided by 2 because 1 complete trade = 1 Buy + 1 Sell
    final_cap = equity_curve[-1]['Capital']
    absolute_roi = ((final_cap - 100000) / 100000) * 100

    days_traded = (pair_data.index[-1] - pair_data.index[0]).days
    if days_traded > 0:
        years = days_traded / 365.25
        annualized_roi = (((final_cap / 100000) ** (1 / years)) - 1) * 100
    else:
        annualized_roi = 0.0

    results.append({
        'Pair': pair_name,
        'P-Value': round(p_val, 4),
        'Trades': trades_made,
        'Final Cap (₹)': round(final_cap, 2),
        'Abs ROI (%)': round(absolute_roi, 2),
        'Yearly ROI (%)': round(annualized_roi, 2)
    })

# ---------------------------------------------------------
# PHASE 3: Summary Report & CSV Export
# ---------------------------------------------------------
print("\n--- PHASE 3: Final Results ---")
if results:
    results_df = pd.DataFrame(results).sort_values(by='Yearly ROI (%)', ascending=False).reset_index(drop=True)
    results_df.to_csv('final_results.csv', index=False)
    print(results_df.to_string())

    # Export the detailed trade log to CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('detailed_trades_log.csv', index=False)
        print("\n💾 Success! Detailed trade ledger saved as 'detailed_trades_log.csv' in your current directory.")
else:
    print("No valid pairs executed trades during this period.")