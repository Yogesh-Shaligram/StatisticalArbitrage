import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import itertools
import json
import os

# --- CONFIGURATION (REAL WORLD) ---
START_DATE = '2022-01-01'
END_DATE = '2025-01-01'
INITIAL_CAPITAL = 1500000
MAX_PAIRS = 3
LEVERAGE = 2.0
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.0
STOP_LOSS_Z = 4.0
TRAIN_MONTHS = 12
TEST_MONTHS = 3
TXN_COST_PCT = 0.001  # 0.1% Slippage + Brokerage

# --- REALISTIC LOT SIZES (Approximate) ---
# You must update these monthly as exchanges revise them
LOT_SIZES = {
    'HDFCBANK.NS': 550, 'ICICIBANK.NS': 700, 'AXISBANK.NS': 625,
    'KOTAKBANK.NS': 400, 'INDUSINDBK.NS': 500, 'FEDERALBNK.NS': 5000,
    'SBIN.NS': 1500, 'CANBK.NS': 2700, 'PNB.NS': 8000,
    'BANKBARODA.NS': 2925, 'UNIONBANK.NS': 5000,
    'TCS.NS': 175, 'INFY.NS': 400, 'HCLTECH.NS': 700, 'WIPRO.NS': 1500,
    'LTIM.NS': 150, 'PERSISTENT.NS': 200, 'COFORGE.NS': 150, 'MPHASIS.NS': 275,
    'M&M.NS': 350, 'MARUTI.NS': 100, 'BAJAJ-AUTO.NS': 125,
    'TVSMOTOR.NS': 350, 'EICHERMOT.NS': 175, 'HEROMOTOCO.NS': 300,
    'UPL.NS': 1300, 'TATACHEM.NS': 550, 'PIIND.NS': 250,
    'SRF.NS': 375, 'AARTIIND.NS': 1000, 'ATUL.NS': 75
}

SECTOR_MAP = {
    'HDFCBANK.NS': 'Bank', 'ICICIBANK.NS': 'Bank', 'AXISBANK.NS': 'Bank',
    'KOTAKBANK.NS': 'Bank', 'INDUSINDBK.NS': 'Bank', 'FEDERALBNK.NS': 'Bank',
    'SBIN.NS': 'Bank', 'CANBK.NS': 'Bank', 'PNB.NS': 'Bank',
    'BANKBARODA.NS': 'Bank', 'UNIONBANK.NS': 'Bank',
    'TCS.NS': 'IT', 'INFY.NS': 'IT', 'HCLTECH.NS': 'IT', 'WIPRO.NS': 'IT',
    'LTIM.NS': 'IT', 'PERSISTENT.NS': 'IT', 'COFORGE.NS': 'IT', 'MPHASIS.NS': 'IT',
    'M&M.NS': 'Auto', 'MARUTI.NS': 'Auto', 'BAJAJ-AUTO.NS': 'Auto',
    'TVSMOTOR.NS': 'Auto', 'EICHERMOT.NS': 'Auto', 'HEROMOTOCO.NS': 'Auto',
    'UPL.NS': 'Chem', 'TATACHEM.NS': 'Chem', 'PIIND.NS': 'Chem',
    'SRF.NS': 'Chem', 'AARTIIND.NS': 'Chem', 'ATUL.NS': 'Chem'
}


class ProductionBacktester:
    def __init__(self, sector_map, start, end, capital):
        self.sector_map = sector_map
        self.tickers = list(sector_map.keys())
        self.capital = capital
        self.current_equity = capital
        self.trade_records = []
        self.state_file = "portfolio_state.json"

        print(f"📥 Downloading data for {len(self.tickers)} stocks...")
        self.data = yf.download(self.tickers, start=start, end=end)['Close'].ffill().dropna(axis=1)

    def run(self):
        dates = self.data.index
        curr_date = dates[0]

        print(f"\n🚀 STARTING PRODUCTION SIMULATION | Capital: ₹{self.capital:,.0f}")
        print(f"⚙️  Enforcing Futures Lot Sizes & Margin Checks")

        daily_equity = pd.Series(np.nan, index=self.data.index)
        daily_equity.iloc[0] = float(self.capital)

        while True:
            train_end = curr_date + pd.DateOffset(months=TRAIN_MONTHS)
            test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
            if test_end > dates[-1]: break

            # 1. Research
            train_df = self.data.loc[curr_date:train_end]
            top_pairs = self.get_top_pairs(train_df, n=MAX_PAIRS)

            # 2. Allocation
            if top_pairs:
                allocation_per_pair = (self.current_equity * LEVERAGE) / len(top_pairs)
            else:
                allocation_per_pair = 0

            # 3. Trade
            test_df = self.data.loc[train_end:test_end]
            segment_pnl_accumulator = pd.Series(0.0, index=test_df.index)

            if top_pairs:
                for p in top_pairs:
                    s1, s2, hr, _ = p
                    pair_pnl_curve, final_pnl = self.trade_pair_real_world(
                        test_df, s1, s2, hr, allocation_per_pair
                    )
                    segment_pnl_accumulator += pair_pnl_curve
                    self.current_equity += final_pnl

            # 4. Update Equity
            prev_idx = daily_equity.index.get_loc(train_end) if train_end in daily_equity.index else \
            daily_equity.index.get_indexer([train_end], method='ffill')[0]
            start_eq = daily_equity.iloc[prev_idx]
            if pd.isna(start_eq): start_eq = self.current_equity

            daily_equity.loc[test_df.index] = start_eq + segment_pnl_accumulator.cumsum()

            curr_date += pd.DateOffset(months=TEST_MONTHS)

        daily_equity = daily_equity.ffill()
        self.generate_report(daily_equity)

    def get_top_pairs(self, df, n):
        pairs = []
        for s1, s2 in itertools.combinations(df.columns, 2):
            if self.sector_map.get(s1) != self.sector_map.get(s2): continue
            try:
                _, p, _ = coint(df[s1], df[s2])
                if p < 0.05:
                    hr = sm.OLS(df[s1], sm.add_constant(df[s2])).fit().params.iloc[1]
                    pairs.append((s1, s2, hr, p))
            except:
                continue
        return sorted(pairs, key=lambda x: x[3])[:n]

    def get_lot_size(self, ticker):
        return LOT_SIZES.get(ticker, 1)  # Default to 1 if missing (Safety)

    def calculate_lots(self, allocation, price_s1, price_s2, hr, s1_n, s2_n):
        """
        Calculates maximum possible lots allowed by capital & risk.
        Ensures S1 Lots and S2 Lots maintain the Hedge Ratio.
        """
        lot_s1 = self.get_lot_size(s1_n)
        lot_s2 = self.get_lot_size(s2_n)

        # Value of 1 Lot of Spread = (P1*Lot1) + (P2*Lot2*HR_adjusted)
        # We need to find 'k' (number of lots) such that Cost < Allocation

        # Ideal Qty S1
        raw_qty_s1 = allocation / (price_s1 + abs(hr) * price_s2)

        # Round to Lot Multiples
        num_lots_s1 = int(raw_qty_s1 / lot_s1)

        if num_lots_s1 < 1:
            return 0, 0  # Capital too low for even 1 lot

        qty_s1 = num_lots_s1 * lot_s1

        # Match S2 Qty based on HR
        raw_qty_s2 = qty_s1 * abs(hr)
        num_lots_s2 = round(raw_qty_s2 / lot_s2)
        qty_s2 = num_lots_s2 * lot_s2

        if qty_s2 == 0: return 0, 0

        return qty_s1, qty_s2

    def save_state(self, pair, status, details):
        """Mock saving state to JSON for crash recovery"""
        state = {
            "pair": pair,
            "status": status,
            "details": details,
            "equity": self.current_equity
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def trade_pair_real_world(self, df, s1_n, s2_n, hr, allocation):
        s1_prices = df[s1_n]
        s2_prices = df[s2_n]

        spread = s1_prices - hr * s2_prices
        z_score = (spread - spread.mean()) / spread.std()

        position = 0
        entry_s1 = 0;
        entry_s2 = 0
        qty_s1 = 0;
        qty_s2 = 0
        realized_pnl = 0
        daily_pnl = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            date = df.index[i]
            z = z_score.iloc[i]
            p1 = s1_prices.iloc[i]
            p2 = s2_prices.iloc[i]

            # --- STOP LOSS LOGIC ---
            if position != 0:
                if abs(z) > STOP_LOSS_Z:
                    # EXIT NOW
                    curr_pnl_s1 = (p1 - entry_s1) * qty_s1 * position
                    curr_pnl_s2 = (entry_s2 - p2) * qty_s2 * position
                    gross_pnl = curr_pnl_s1 + curr_pnl_s2
                    exit_cost = (p1 * qty_s1 + p2 * qty_s2) * TXN_COST_PCT
                    net_loss = gross_pnl - exit_cost

                    realized_pnl += net_loss
                    daily_pnl.iloc[i] = net_loss

                    self.log_trade(date, f"{s1_n}/{s2_n}", "STOP (Z)", f"Z:{z:.1f} Lots:{qty_s1}/{qty_s2}", net_loss)
                    self.save_state(f"{s1_n}/{s2_n}", "FLAT", "Stop Hit")

                    position = 0;
                    qty_s1 = 0;
                    qty_s2 = 0
                    continue

            # --- ENTRY LOGIC ---
            if position == 0:
                if abs(z) > ENTRY_THRESHOLD:
                    # Calculate LOTS based on Real Constraints
                    q1, q2 = self.calculate_lots(allocation, p1, p2, hr, s1_n, s2_n)

                    if q1 > 0 and q2 > 0:
                        # Determine Direction
                        if z < -ENTRY_THRESHOLD:
                            position = 1  # Long Spread
                            entry_s1, entry_s2 = p1, p2
                            qty_s1, qty_s2 = q1, q2
                            action = "LONG"
                        else:
                            position = -1  # Short Spread
                            entry_s1, entry_s2 = p1, p2
                            qty_s1, qty_s2 = q1, q2
                            action = "SHORT"

                        # Apply Entry Cost
                        entry_cost = (p1 * qty_s1 + p2 * qty_s2) * TXN_COST_PCT
                        realized_pnl -= entry_cost

                        self.log_trade(date, f"{s1_n}/{s2_n}", action,
                                       f"Lots: {qty_s1}/{qty_s2} (Cost: {entry_cost:.0f})", -entry_cost)
                        self.save_state(f"{s1_n}/{s2_n}", "OPEN", f"{action} {qty_s1}/{qty_s2}")
                    else:
                        # Failed Lot Check (Capital too low for 1 lot)
                        # We don't log this to avoid spamming CSV, but in real time we would alert user
                        pass

            # --- EXIT LOGIC ---
            elif position != 0:
                # Mean Reversion Target
                if (position == 1 and z > -EXIT_THRESHOLD) or (position == -1 and z < EXIT_THRESHOLD):
                    pnl_s1 = (p1 - entry_s1) * qty_s1 * position
                    pnl_s2 = (entry_s2 - p2) * qty_s2 * position
                    gross_pnl = pnl_s1 + pnl_s2
                    exit_cost = (p1 * qty_s1 + p2 * qty_s2) * TXN_COST_PCT
                    net_pnl = gross_pnl - exit_cost

                    realized_pnl += net_pnl
                    daily_pnl.iloc[i] = net_pnl

                    self.log_trade(date, f"{s1_n}/{s2_n}", "EXIT", f"Target Hit. PnL: {net_pnl:.0f}", net_pnl)
                    self.save_state(f"{s1_n}/{s2_n}", "FLAT", "Profit Taken")

                    position = 0;
                    qty_s1 = 0;
                    qty_s2 = 0

        return daily_pnl, realized_pnl

    def log_trade(self, date, pair, action, details, pnl):
        self.trade_records.append({
            'Date': date, 'Pair': pair, 'Action': action, 'Details': details,
            'Realized_PnL': round(pnl, 2), 'Portfolio_Equity': round(self.current_equity + pnl, 2)
        })

    def generate_report(self, equity_curve):
        trade_df = pd.DataFrame(self.trade_records)
        trade_df.to_csv("Live_Simulation_Trades.csv", index=False)

        total_txn_costs = sum(abs(r['Realized_PnL']) for r in self.trade_records if "Cost" in str(r['Details']))

        final_eq = equity_curve.iloc[-1]
        total_ret = (final_eq - self.capital) / self.capital * 100
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        cagr = ((final_eq / self.capital) ** (365 / days) - 1) * 100

        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak * 100
        max_dd = dd.min()
        sharpe = (equity_curve.pct_change().mean() / equity_curve.pct_change().std()) * np.sqrt(252)

        print("\n" + "=" * 60)
        print(f"PRODUCTION SIMULATION REPORT (Initial: ₹{self.capital:,.0f})")
        print("=" * 60)
        print(f"Final Equity:      ₹{final_eq:,.2f}")
        print(f"Total Return:      {total_ret:.2f}%")
        print(f"CAGR:              {cagr:.2f}%")
        print(f"Txn Costs Paid:    ₹{total_txn_costs:,.2f}")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Max Drawdown:      {max_dd:.2f}%")
        print("=" * 60)

        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, color='navy', label='Futures Portfolio')
        plt.title(f"Production Strategy (Lots Enforced) | CAGR: {cagr:.1f}%")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()
        plt.savefig("Production_Performance.png")
        print("Graph saved ")
        print("Trade Log saved")
        print("State file created")


if __name__ == "__main__":
    bot = ProductionBacktester(SECTOR_MAP, START_DATE, END_DATE, INITIAL_CAPITAL)
    bot.run()