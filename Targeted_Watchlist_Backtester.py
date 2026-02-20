import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PREPARATION: The Isolated 16-Pair Universe
# ---------------------------------------------------------
# Hardcoding the exact pairs from the leaderboard
pairs = [
    ('AXISBANK.NS', 'BANKBARODA.NS'),
    ('KOTAKBANK.NS', 'BANKBARODA.NS'),
    ('DRREDDY.NS', 'LUPIN.NS'),
    ('DABUR.NS', 'GODREJCP.NS'),
    ('KOTAKBANK.NS', 'PNB.NS'),
    ('MARUTI.NS', 'TVSMOTOR.NS'),
    ('JSWSTEEL.NS', 'COALINDIA.NS'),
    ('HDFCBANK.NS', 'FEDERALBNK.NS'),
    ('M&M.NS', 'EICHERMOT.NS'),
    ('KOTAKBANK.NS', 'FEDERALBNK.NS'),
    ('BAJFINANCE.NS', 'BAJAJFINSV.NS'),
    ('TCS.NS', 'PERSISTENT.NS'),
    ('HDFCBANK.NS', 'ICICIBANK.NS'),
    ('BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'),
    ('RELIANCE.NS', 'TATAPOWER.NS'),
    ('RELIANCE.NS', 'ONGC.NS')
]

all_tickers = set()
for asset1, asset2 in pairs:
    all_tickers.update([asset1, asset2])

print(f"Loaded {len(pairs)} target pairs. Downloading 5 years of data...")

end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)  # 5 Years Total
split_date = start_date + timedelta(days=int(3.5 * 365))  # 3.5 Years In-Sample

data = yf.download(list(all_tickers), start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'),
                   progress=False)['Close']
data = data.ffill().dropna()

# Split the data
data_is = data.loc[:split_date]  # In-Sample (Training)
data_oos = data.loc[split_date:]  # Out-of-Sample (Live Testing)

print(f"\nTraining (In-Sample) Period: {data_is.index[0].date()} to {data_is.index[-1].date()}")
print(f"Testing (Out-of-Sample) Period: {data_oos.index[0].date()} to {data_oos.index[-1].date()}")

# ---------------------------------------------------------
# PHASE 1: Scan IS Data for Cointegration & Train Ratio
# ---------------------------------------------------------
print("\n--- PHASE 1: Training Models on In-Sample Data ---")
trained_pairs = []

for asset1, asset2 in pairs:
    if asset1 in data_is.columns and asset2 in data_is.columns:
        pair_data_is = data_is[[asset1, asset2]].dropna()
        if len(pair_data_is) > 100:
            score, p_value, _ = ts.coint(pair_data_is[asset1], pair_data_is[asset2])

            if p_value < 0.05:
                # Calculate the exact ratio during the training period
                model = sm.OLS(pair_data_is[asset1], pair_data_is[asset2]).fit()
                trained_ratio = model.params.iloc[0]
                trained_pairs.append((asset1, asset2, p_value, trained_ratio))

print(f"✅ Found {len(trained_pairs)} fundamentally solid pairs in the training data.")

# ---------------------------------------------------------
# PHASE 2: Trade OOS Data (Forward Walk)
# ---------------------------------------------------------
print("--- PHASE 2: Executing Trades on Unseen OOS Data ---")
results = []
all_trades = []
ENTRY_Z = 1.75
EXIT_Z = 0.0

for asset1, asset2, p_val, ratio in trained_pairs:
    pair_name = f"{asset1.replace('.NS', '')} / {asset2.replace('.NS', '')}"
    pair_data_oos = data_oos[[asset1, asset2]].dropna().copy()

    if len(pair_data_oos) < 50:
        continue  # Skip if missing recent data

    # Apply the TRAINED ratio to the UNSEEN data
    pair_data_oos['Spread'] = pair_data_oos[asset1] - (ratio * pair_data_oos[asset2])
    pair_data_oos['SMA'] = pair_data_oos['Spread'].rolling(window=20).mean()
    pair_data_oos['StdDev'] = pair_data_oos['Spread'].rolling(window=20).std()
    pair_data_oos['Z_Score'] = (pair_data_oos['Spread'] - pair_data_oos['SMA']) / pair_data_oos['StdDev']
    pair_data_oos = pair_data_oos.dropna()

    position = 0
    units = 0
    cash = 100000.0
    equity_curve = [{'Date': pair_data_oos.index[0], 'Capital': cash}]
    trades_for_pair = 0

    for index, row in pair_data_oos.iterrows():
        z_score = row['Z_Score']
        p1 = row[asset1]
        p2 = row[asset2]

        # Entry Signals
        if position == 0:
            if z_score < -ENTRY_Z:
                units = int(cash // p1)
                if units > 0:
                    trade_cost = units * p1
                    cash -= trade_cost
                    position = 1
                    all_trades.append(
                        {'Date': index, 'Pair': pair_name, 'Asset': asset1, 'Action': 'BUY', 'Price': round(p1, 2),
                         'Qty': units})

            elif z_score > ENTRY_Z:
                units = int(cash // p2)
                if units > 0:
                    trade_cost = units * p2
                    cash -= trade_cost
                    position = 2
                    all_trades.append(
                        {'Date': index, 'Pair': pair_name, 'Asset': asset2, 'Action': 'BUY', 'Price': round(p2, 2),
                         'Qty': units})

        # Exit Signals
        elif position == 1:
            if z_score > EXIT_Z:
                cash += (units * p1)
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset': asset1, 'Action': 'SELL', 'Price': round(p1, 2),
                     'Qty': units})
                equity_curve.append({'Date': index, 'Capital': cash})
                position = 0
                units = 0
                trades_for_pair += 1

        elif position == 2:
            if z_score < EXIT_Z:
                cash += (units * p2)
                all_trades.append(
                    {'Date': index, 'Pair': pair_name, 'Asset': asset2, 'Action': 'SELL', 'Price': round(p2, 2),
                     'Qty': units})
                equity_curve.append({'Date': index, 'Capital': cash})
                position = 0
                units = 0
                trades_for_pair += 1

    # End of OOS period cleanup (sell open positions at final market price)
    if position == 1:
        cash += (units * pair_data_oos.iloc[-1][asset1])
    elif position == 2:
        cash += (units * pair_data_oos.iloc[-1][asset2])

    # OOS Performance Metrics
    absolute_roi = ((cash - 100000) / 100000) * 100
    days_traded = (pair_data_oos.index[-1] - pair_data_oos.index[0]).days

    annualized_roi = 0.0
    if days_traded > 0 and cash > 0:
        years = days_traded / 365.25
        annualized_roi = (((cash / 100000) ** (1 / years)) - 1) * 100

    results.append({
        'Pair': pair_name,
        'IS P-Value': round(p_val, 4),
        'OOS Trades': trades_for_pair,
        'OOS ROI (%)': round(absolute_roi, 2),
        'OOS Yearly (%)': round(annualized_roi, 2)
    })

# ---------------------------------------------------------
# PHASE 3: Summary Report & CSV Export
# ---------------------------------------------------------
print("\n--- PHASE 3: True Out-of-Sample Results ---")
if results:
    df_results = pd.DataFrame(results)
    df_traded = df_results[df_results['OOS Trades'] > 0]

    if not df_traded.empty:
        # Sort by actual live-testing performance
        df_sorted = df_traded.sort_values(by='OOS Yearly (%)', ascending=False).reset_index(drop=True)
        print(df_sorted.to_string())
    else:
        print("Pairs were cointegrated in training, but no trades triggered in the testing phase.")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('oos_validation_trades.csv', index=False)
        print("\n💾 Success! True validation ledger saved as 'oos_validation_trades.csv'.")
else:
    print("No valid pairs passed the training phase.")