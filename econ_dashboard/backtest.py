import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = datetime.now() - relativedelta(years=10)
# Fetch 11 years of data to have a buffer for initial calculations
DATA_FETCH_START_DATE = START_DATE - relativedelta(years=1)
END_DATE = datetime.now()

INITIAL_INVESTMENT = 10000.00

# --- Strategy Definitions (Copied from the dashboard) ---

INDICATOR_SERIES = {
    'Industrial Production Index': ('INDPRO', True),
    'Chicago Fed National Activity Index': ('CFNAI', True),
    'Consumer Sentiment Index': ('UMCSENT', True),
    'New Privately-Owned Housing Units Started': ('HOUST', True),
    'Initial Jobless Claims': ('ICSA', False),
    'Durable Goods Orders (MoM %)': ('DGORDER', True),
    'Retail Sales (MoM %)': ('RSXFS', True),
    'Building Permits (Thousands)': ('PERMIT', True),
    'Unemployment Rate (%)': ('UNRATE', False),
    'Average Weekly Hours, Manufacturing': ('AWHMAN', True),
}

CREDIT_SPREAD_SERIES = {
    'High-Yield Spread (OAS)': ('BAMLH0A0HYM2', False),
    'Baa Corp Bond Yield': ('DBAA', False),
    '10-Year Treasury Yield': ('DGS10', False)
}

STRATEGY_ETFS = {
    "Optimistic Outlook (Pro-Risk)": {"Equities": 0.65, "Bonds": 0.20, "Commodities": 0.10, "Cash": 0.05},
    "Cautiously Optimistic (Balanced)": {"Equities": 0.55, "Bonds": 0.30, "Real Estate": 0.10, "Cash": 0.05},
    "Neutral / Cautious (Defensive)": {"Equities": 0.40, "Bonds": 0.45, "Alternatives": 0.10, "Cash": 0.05},
    "Pessimistic Outlook (Risk-Off)": {"Equities": 0.25, "Bonds": 0.55, "Gold": 0.15, "Cash": 0.05}
}

ASSET_CLASS_MAP = {
    "Equities": "VTI",
    "Bonds": "AGG",
    "Commodities": "DBC",
    "Real Estate": "VNQ",
    "Alternatives": "PSP",
    "Gold": "GLD",
    "Cash": "BIL"
}

BENCHMARK_TICKER = "SPY"

def fetch_all_data():
    """Fetches all economic and ETF data for the backtest period."""
    print("--- Fetching all required historical data. This may take a minute... ---")
    
    all_series_ids = [v[0] for v in INDICATOR_SERIES.values()] + [v[0] for v in CREDIT_SPREAD_SERIES.values()]
    try:
        econ_data = web.DataReader(all_series_ids, 'fred', DATA_FETCH_START_DATE, END_DATE, api_key=FRED_API_KEY)
        print("Successfully fetched economic data from FRED.")
    except Exception as e:
        print(f"ERROR: Could not fetch economic data from FRED. {e}")
        return None, None

    econ_data['Baa - 10-Year Treasury'] = econ_data['DBAA'] - econ_data['DGS10']
    
    all_etfs = list(set(ASSET_CLASS_MAP.values()) | {BENCHMARK_TICKER})
    try:
        # --- MODIFIED: Changed from 'Adj Close' to 'Close' ---
        etf_data = yf.download(all_etfs, start=DATA_FETCH_START_DATE, end=END_DATE)['Close']
        
        etf_monthly_prices = etf_data.resample('M').last()
        etf_monthly_returns = etf_monthly_prices.pct_change().dropna()
        print("Successfully fetched and processed ETF data using 'Close' price.")
    except Exception as e:
        print(f"ERROR: Could not fetch ETF data from yfinance. {e}")
        return None, None
        
    return econ_data, etf_monthly_returns

def generate_recommendation_for_date(current_date, econ_data):
    """Replicates the dashboard's scoring logic for a specific point in time."""
    data_snapshot = econ_data[econ_data.index < current_date]
    if data_snapshot.empty:
        return "Neutral / Cautious (Defensive)" 
    
    score = 0
    # Score Indicators
    for name, (series_id, is_good_high) in INDICATOR_SERIES.items():
        if series_id in data_snapshot.columns:
            series = data_snapshot[series_id].dropna()
            if not series.empty:
                latest = series.iloc[-1]
                mean = series.mean()
                if (is_good_high and latest > mean) or (not is_good_high and latest < mean):
                    score += 1
                else:
                    score -= 1
    
    # Score Spreads
    spreads_to_score = {'BAMLH0A0HYM2': False, 'Baa - 10-Year Treasury': False}
    for series_id, is_good_high in spreads_to_score.items():
        if series_id in data_snapshot.columns:
             series = data_snapshot[series_id].dropna()
             if not series.empty:
                latest = series.iloc[-1]
                mean = series.mean()
                if not is_good_high and latest < mean:
                    score += 2 
                else:
                    score -= 2
    
    if score >= 8: return "Optimistic Outlook (Pro-Risk)"
    elif score >= 2: return "Cautiously Optimistic (Balanced)"
    elif score > -5: return "Neutral / Cautious (Defensive)"
    else: return "Pessimistic Outlook (Risk-Off)"

def run_backtest(econ_data, etf_returns):
    """Main backtesting loop."""
    print("\n--- Running Backtest ---")
    
    test_period_returns = etf_returns[etf_returns.index >= START_DATE]
    
    strategy_values = []
    benchmark_values = []
    
    strategy_value = INITIAL_INVESTMENT
    benchmark_value = INITIAL_INVESTMENT
    
    last_strategy = ""

    for month_end_date, returns_for_month in test_period_returns.iterrows():
        strategy_name = generate_recommendation_for_date(month_end_date, econ_data)
        
        if strategy_name != last_strategy:
            print(f"{month_end_date.strftime('%Y-%m-%d')}: Strategy set to -> {strategy_name}")
            last_strategy = strategy_name
            
        monthly_strategy_return = 0
        allocation = STRATEGY_ETFS[strategy_name]
        for asset_class, weight in allocation.items():
            ticker = ASSET_CLASS_MAP[asset_class]
            asset_return = returns_for_month.get(ticker, 0)
            if pd.isna(asset_return): asset_return = 0
            monthly_strategy_return += weight * asset_return
        
        strategy_value *= (1 + monthly_strategy_return)
        strategy_values.append(strategy_value)
        
        benchmark_return = returns_for_month.get(BENCHMARK_TICKER, 0)
        if pd.isna(benchmark_return): benchmark_return = 0
        benchmark_value *= (1 + benchmark_return)
        benchmark_values.append(benchmark_value)

    strategy_portfolio = pd.DataFrame({'Value': strategy_values}, index=test_period_returns.index)
    benchmark_portfolio = pd.DataFrame({'Value': benchmark_values}, index=test_period_returns.index)
    
    return strategy_portfolio, benchmark_portfolio

def calculate_performance_metrics(portfolio):
    """Calculates key performance metrics for a given portfolio timeseries."""
    if portfolio.empty:
        return {"Final Value": 0, "Total Return": 0, "CAGR": 0, "Sharpe Ratio": 0}
        
    total_return = (portfolio.iloc[-1] / INITIAL_INVESTMENT) - 1
    num_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    cagr = ((portfolio.iloc[-1] / INITIAL_INVESTMENT) ** (1/num_years)) - 1
    
    monthly_returns = portfolio['Value'].pct_change().dropna()
    if monthly_returns.std() == 0:
        sharpe_ratio = np.inf if monthly_returns.mean() > 0 else -np.inf
    else:
        sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    
    return {
        "Final Value": portfolio.iloc[-1].values[0],
        "Total Return": total_return.values[0],
        "CAGR": cagr.values[0],
        "Sharpe Ratio": sharpe_ratio
    }

def plot_results(strategy_df, benchmark_df):
    """Plots the performance of the strategy vs. the benchmark."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(strategy_df.index, strategy_df['Value'], label='Strategy Portfolio', color='royalblue', linewidth=2)
    ax.plot(benchmark_df.index, benchmark_df['Value'], label='S&P 500 Benchmark (SPY)', color='grey', linestyle='--')
    
    ax.set_title('Strategy Backtest vs. S&P 500 Benchmark (10-Year Performance)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=12)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.figtext(0.5, 0.01, 'Based on monthly rebalancing of a $10,000 initial investment.', ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    if not FRED_API_KEY:
        print("🛑 ERROR: FRED_API_KEY not found in .env file. Please add it.")
    else:
        econ_data, etf_returns = fetch_all_data()
        
        if econ_data is not None and etf_returns is not None:
            strategy_portfolio, benchmark_portfolio = run_backtest(econ_data, etf_returns)
            
            strategy_metrics = calculate_performance_metrics(strategy_portfolio)
            benchmark_metrics = calculate_performance_metrics(benchmark_portfolio)
            
            print("\n--- Backtest Results ---")
            print("-" * 50)
            print(f"| Metric         | {'Strategy Portfolio':<20} | {'S&P 500 Benchmark':<20} |")
            print("-" * 50)
            print(f"| Final Value    | ${strategy_metrics['Final Value']:<19,.2f} | ${benchmark_metrics['Final Value']:<19,.2f} |")
            print(f"| Total Return   | {strategy_metrics['Total Return']:<19.2%} | {benchmark_metrics['Total Return']:<19.2%} |")
            print(f"| CAGR           | {strategy_metrics['CAGR']:<19.2%} | {benchmark_metrics['CAGR']:<19.2%} |")
            print(f"| Sharpe Ratio   | {strategy_metrics['Sharpe Ratio']:<19.2f} | {benchmark_metrics['Sharpe Ratio']:<19.2f} |")
            print("-" * 50)
            
            plot_results(strategy_portfolio, benchmark_portfolio)