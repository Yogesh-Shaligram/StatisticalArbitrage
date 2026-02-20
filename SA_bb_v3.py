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
# PREPARATION: The Nifty 200 Sector Universe
# ---------------------------------------------------------
sectors = {
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS'],
    'Banks': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'PNB.NS',
              'BANKBARODA.NS', 'FEDERALBNK.NS'],
    'Auto': ['M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS'],
    'Pharma_Health': ['SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'FMCG_Retail': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS', 'DABUR.NS',
                    'GODREJCP.NS'],
    'Metals_Mining': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS', 'VEDL.NS'],
    'Energy_Oil': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS']
}

pairs = []
all_tickers = set()

for sector, stocks in sectors.items():
    sector_pairs = list(itertools.combinations(stocks, 2))
    pairs.extend(sector_pairs)
    all_tickers.update(stocks)

print(f"Generated {len(pairs)} pairs. Downloading 5 years of data...")
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)
split_date = start_date + timedelta(days=int(3.5 * 365))

data = yf.download(list(all_tickers), start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'),
                   progress=False)['Close']
data = data.ffill().dropna()

data_is = data.loc[:split_date]  # Training Data
data_oos = data.loc[split_date:]  # Testing Data

# ---------------------------------------------------------
# PHASE 1: Scan IS Data & Train Ratio
# ---------------------------------------------------------
print("\n--- PHASE 1: Training Models on In-Sample Data ---")
trained_pairs = []

for asset1, asset2 in pairs:
    if asset1 in data_is.columns and asset2 in data_is.columns:
        pair_data_is = data_is[[asset1, asset2]].dropna()
        if len(pair_data_is) > 100:
            score, p_value, _ = ts.coint(pair_data_is[asset1], pair_data_is[asset2])
            if p_value < 0.05:
                model = sm.OLS(pair_data_is[asset1], pair_data_is[asset2]).fit()
                trained_ratio = model.params.iloc[0]
                trained_pairs.append((asset1, asset2, p_value, trained_ratio))

print(f"✅ Found {len(trained_pairs)} fundamentally solid pairs.")

# ---------------------------------------------------------
# PHASE 2: Market-Neutral Execution & Stop Loss (OOS Data)
# ---------------------------------------------------------
print("--- PHASE 2: Executing Long/Short Trades on Unseen Data ---")
results = []
all_trades = []

ENTRY_Z = 1.75
EXIT_Z = 0.0
STOP_Z = 3.0  # The Disaster Stop-Loss
FEE_PER_TRADE = 100.0  # Realistic ₹100 fee per round-trip

for asset1, asset2, p_val, ratio in trained_pairs:
    pair_name = f"{asset1.replace('.NS', '')} / {asset2.replace('.NS', '')}"
    pair_data_oos = data_oos[[asset1, asset2]].dropna().copy()

    if len(pair_data_oos) < 50:
        continue

    pair_data_oos['Spread'] = pair_data_oos[asset1] - (ratio * pair_data_oos[asset2])
    pair_data_oos['SMA'] = pair_data_oos['Spread'].rolling(window=20).mean()
    pair_data_oos['StdDev'] = pair_data_oos['Spread'].rolling(window=20).std()
    pair_data_oos['Z_Score'] = (pair_data_oos['Spread'] - pair_data_oos['SMA']) / pair_data_oos['StdDev']
    pair_data_oos = pair_data_oos.dropna()

    position = 0
    units1 = 0;
    units2 = 0
    entry_p1 = 0.0;
    entry_p2 = 0.0
    capital = 100000.0

    equity_curve = [{'Date': pair_data_oos.index[0], 'Capital': capital}]
    trades_for_pair = 0

    for index, row in pair_data_oos.iterrows():
        z_score = row['Z_Score']
        p1 = row[asset1]
        p2 = row[asset2]

        # 1. Entry Signals (50/50 Capital Allocation)
        if position == 0:
            if z_score < -ENTRY_Z:
                # LONG SPREAD: Buy A1, Short A2
                units1 = int((capital * 0.5) // p1)
                units2 = int((capital * 0.5) // p2)

                if units1 > 0 and units2 > 0:
                    entry_p1 = p1;
                    entry_p2 = p2
                    position = 1
                    all_trades.append(
                        {'Date': index, 'Pair': pair_name, 'Action': 'OPEN LONG SPREAD', 'A1_Price': round(p1, 2),
                         'A2_Price': round(p2, 2)})

            elif z_score > ENTRY_Z:
                # SHORT SPREAD: Short A1, Buy A2
                units1 = int((capital * 0.5) // p1)
                units2 = int((capital * 0.5) // p2)

                if units1 > 0 and units2 > 0:
                    entry_p1 = p1;
                    entry_p2 = p2
                    position = -1
                    all_trades.append(
                        {'Date': index, 'Pair': pair_name, 'Action': 'OPEN SHORT SPREAD', 'A1_Price': round(p1, 2),
                         'A2_Price': round(p2, 2)})

        # 2. Exit Signals (Mean Reversion OR Stop-Loss)
        elif position == 1:
            if z_score > EXIT_Z or z_score < -STOP_Z:
                profit_a1 = (p1 - entry_p1) * units1  # Long profit
                profit_a2 = (entry_p2 - p2) * units2  # Short profit
                net_profit = profit_a1 + profit_a2 - FEE_PER_TRADE

                capital += net_profit
                reason = "TAKE PROFIT" if z_score > EXIT_Z else "STOP LOSS"
                all_trades.append({'Date': index, 'Pair': pair_name, 'Action': f'CLOSE SPREAD ({reason})',
                                   'Net_Profit': round(net_profit, 2)})

                equity_curve.append({'Date': index, 'Capital': capital})
                position = 0;
                trades_for_pair += 1

        elif position == -1:
            if z_score < EXIT_Z or z_score > STOP_Z:
                profit_a1 = (entry_p1 - p1) * units1  # Short profit
                profit_a2 = (p2 - entry_p2) * units2  # Long profit
                net_profit = profit_a1 + profit_a2 - FEE_PER_TRADE

                capital += net_profit
                reason = "TAKE PROFIT" if z_score < EXIT_Z else "STOP LOSS"
                all_trades.append({'Date': index, 'Pair': pair_name, 'Action': f'CLOSE SPREAD ({reason})',
                                   'Net_Profit': round(net_profit, 2)})

                equity_curve.append({'Date': index, 'Capital': capital})
                position = 0;
                trades_for_pair += 1

    # End of OOS cleanup (Force close any open positions at the very end)
    if position == 1:
        net_profit = ((pair_data_oos.iloc[-1][asset1] - entry_p1) * units1) + (
                    (entry_p2 - pair_data_oos.iloc[-1][asset2]) * units2) - FEE_PER_TRADE
        capital += net_profit
    elif position == -1:
        net_profit = ((entry_p1 - pair_data_oos.iloc[-1][asset1]) * units1) + (
                    (pair_data_oos.iloc[-1][asset2] - entry_p2) * units2) - FEE_PER_TRADE
        capital += net_profit

    absolute_roi = ((capital - 100000) / 100000) * 100
    days_traded = (pair_data_oos.index[-1] - pair_data_oos.index[0]).days
    annualized_roi = 0.0
    if days_traded > 0 and capital > 0:
        years = days_traded / 365.25
        annualized_roi = (((capital / 100000) ** (1 / years)) - 1) * 100

    results.append({
        'Pair': pair_name,
        'OOS Trades': trades_for_pair,
        'Final Cap (₹)': round(capital, 2),
        'OOS ROI (%)': round(absolute_roi, 2),
        'Yearly ROI (%)': round(annualized_roi, 2)
    })

# ---------------------------------------------------------
# PHASE 3: Summary Report
# ---------------------------------------------------------
print("\n--- PHASE 3: True Out-of-Sample Results ---")
if results:
    df_results = pd.DataFrame(results)
    df_traded = df_results[df_results['OOS Trades'] > 0]

    if not df_traded.empty:
        df_sorted = df_traded.sort_values(by='Yearly ROI (%)', ascending=False).reset_index(drop=True)
        print(df_sorted.to_string())
    else:
        print("No trades triggered in the testing phase.")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('true_arbitrage_ledger.csv', index=False)
        print("\n💾 Success! Ledger saved as 'true_arbitrage_ledger.csv'.")