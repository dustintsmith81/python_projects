import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from scipy.optimize import minimize

# --- Configuration ---
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = datetime.now() - relativedelta(years=10)
DATA_FETCH_START_DATE = START_DATE - relativedelta(years=1)
END_DATE = datetime.now()

INITIAL_INVESTMENT = 10000.00
REBALANCING_PERIODS_TO_TEST = [1, 2, 3, 6]
CAGR_TARGET_PREMIUM = 0.05  # Target CAGR is benchmark CAGR + this premium

# --- Data & Strategy Definitions ---
INDICATOR_SERIES = {
    'Industrial Production Index': ('INDPRO', True), 'Chicago Fed National Activity Index': ('CFNAI', True),
    'Consumer Sentiment Index': ('UMCSENT', True), 'New Privately-Owned Housing Units Started': ('HOUST', True),
    'Initial Jobless Claims': ('ICSA', False), 'Durable Goods Orders (MoM %)': ('DGORDER', True),
    'Retail Sales (MoM %)': ('RSXFS', True), 'Building Permits (Thousands)': ('PERMIT', True),
    'Unemployment Rate (%)': ('UNRATE', False), 'Average Weekly Hours, Manufacturing': ('AWHMAN', True),
}
CREDIT_SPREAD_SERIES = {
    'High-Yield Spread (OAS)': ('BAMLH0A0HYM2', False), 'Baa Corp Bond Yield': ('DBAA', False),
    '10-Year Treasury Yield': ('DGS10', False)
}
ASSET_CLASSES = ["Equities", "Bonds", "Commodities", "Real Estate", "Alternatives", "Gold", "Cash"]
STRATEGY_NAMES = ["Optimistic Outlook (Pro-Risk)", "Cautiously Optimistic (Balanced)", "Neutral / Cautious (Defensive)", "Pessimistic Outlook (Risk-Off)"]
ASSET_CLASS_MAP = {"Equities": "VTI", "Bonds": "AGG", "Commodities": "DBC", "Real Estate": "VNQ", "Alternatives": "PSP", "Gold": "GLD", "Cash": "BIL"}
BENCHMARK_TICKER = "SPY"
ORIGINAL_STRATEGY_ETFS = {
    "Optimistic Outlook (Pro-Risk)": {"Equities": 0.65, "Bonds": 0.20, "Commodities": 0.10, "Cash": 0.05},
    "Cautiously Optimistic (Balanced)": {"Equities": 0.55, "Bonds": 0.30, "Real Estate": 0.10, "Cash": 0.05},
    "Neutral / Cautious (Defensive)": {"Equities": 0.40, "Bonds": 0.45, "Alternatives": 0.10, "Cash": 0.05},
    "Pessimistic Outlook (Risk-Off)": {"Equities": 0.25, "Bonds": 0.55, "Gold": 0.15, "Cash": 0.05}
}

# --- Global variables ---
ECON_DATA = None
ETF_RETURNS = None
TARGET_CAGR = None

def fetch_all_data():
    global ECON_DATA, ETF_RETURNS
    print("--- Fetching all required historical data. This may take a minute... ---")
    all_series_ids = [v[0] for v in INDICATOR_SERIES.values()] + [v[0] for v in CREDIT_SPREAD_SERIES.values()]
    ECON_DATA = web.DataReader(all_series_ids, 'fred', DATA_FETCH_START_DATE, END_DATE, api_key=FRED_API_KEY)
    ECON_DATA['Baa - 10-Year Treasury'] = ECON_DATA['DBAA'] - ECON_DATA['DGS10']
    all_etfs = list(set(ASSET_CLASS_MAP.values()) | {BENCHMARK_TICKER})
    etf_data = yf.download(all_etfs, start=DATA_FETCH_START_DATE, end=END_DATE)['Close']
    ETF_RETURNS = etf_data.pct_change()
    print("--- Data fetching complete ---")

def generate_recommendation_for_date(current_date, econ_data):
    data_snapshot = econ_data[econ_data.index < current_date]
    if data_snapshot.empty: return "Neutral / Cautious (Defensive)"
    score = 0
    for series_id, is_good_high in INDICATOR_SERIES.values():
        if series_id in data_snapshot.columns:
            series = data_snapshot[series_id].dropna()
            if not series.empty: score += 1 if (is_good_high and series.iloc[-1] > series.mean()) or (not is_good_high and series.iloc[-1] < series.mean()) else -1
    spreads_to_score = {'BAMLH0A0HYM2': False, 'Baa - 10-Year Treasury': False}
    for series_id, is_good_high in spreads_to_score.items():
        if series_id in data_snapshot.columns:
            series = data_snapshot[series_id].dropna()
            if not series.empty: score += 2 if not is_good_high and series.iloc[-1] < series.mean() else -2
    if score >= 8: return STRATEGY_NAMES[0]
    elif score >= 2: return STRATEGY_NAMES[1]
    elif score > -5: return STRATEGY_NAMES[2]
    else: return STRATEGY_NAMES[3]

def run_backtest_with_weights(strategy_weights, rebalancing_period):
    trading_days_returns = ETF_RETURNS[ETF_RETURNS.index >= START_DATE].fillna(0)
    daily_weights = pd.DataFrame(index=trading_days_returns.index, columns=ASSET_CLASS_MAP.values(), dtype=np.float64)
    rebalance_dates = pd.date_range(start=START_DATE, end=END_DATE, freq=f'{rebalancing_period}MS')
    for date in rebalance_dates:
        strategy_name = generate_recommendation_for_date(date, ECON_DATA)
        allocation = strategy_weights[strategy_name]
        period_weights = {ASSET_CLASS_MAP[asset]: weight for asset, weight in allocation.items()}
        if date in daily_weights.index:
            for ticker, weight in period_weights.items(): daily_weights.loc[date, ticker] = weight
        else:
            next_trading_day = daily_weights.index[daily_weights.index > date]
            if not next_trading_day.empty:
                for ticker, weight in period_weights.items(): daily_weights.loc[next_trading_day[0], ticker] = weight
    daily_weights = daily_weights.ffill().fillna(0)
    portfolio_daily_returns = (trading_days_returns[daily_weights.columns] * daily_weights).sum(axis=1)
    portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    return pd.DataFrame({'Value': portfolio_cumulative_returns * INITIAL_INVESTMENT})

def calculate_sharpe_ratio(portfolio):
    if portfolio.empty or len(portfolio) < 2: return -100
    daily_returns = portfolio['Value'].pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0: return -100
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

def calculate_cagr(portfolio):
    if portfolio.empty: return -1
    final_value = portfolio['Value'].iloc[-1]
    num_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    if num_years <=0: return -1
    return ((final_value / INITIAL_INVESTMENT) ** (1/num_years)) - 1

def cagr_constraint(weights_flat, rebalancing_period, strategy_map):
    portfolio = run_backtest_with_weights(construct_strategy_weights(weights_flat, strategy_map), rebalancing_period)
    cagr = calculate_cagr(portfolio)
    return cagr - TARGET_CAGR

def objective_function(weights_flat, rebalancing_period, strategy_map):
    portfolio = run_backtest_with_weights(construct_strategy_weights(weights_flat, strategy_map), rebalancing_period)
    sharpe = calculate_sharpe_ratio(portfolio)
    print(f"  Testing weights for {rebalancing_period}-month period... Sharpe: {sharpe:.4f}")
    return -sharpe

def construct_strategy_weights(weights_flat, strategy_map):
    strategy_weights = {}
    i = 0
    for strategy_name, assets in strategy_map.items():
        strategy_weights[strategy_name] = {}
        for asset_class in assets:
            strategy_weights[strategy_name][asset_class] = weights_flat[i]
            i += 1
    return strategy_weights

def optimize_strategy(rebalancing_period):
    print(f"\n--- Optimizing for {rebalancing_period}-Month Rebalancing Period ---")
    strategy_asset_map = {name: list(assets.keys()) for name, assets in ORIGINAL_STRATEGY_ETFS.items()}
    initial_weights, bounds = [], []
    
    # --- NEW: Logical constraints based on strategy name ---
    for name in STRATEGY_NAMES:
        assets = strategy_asset_map[name]
        for asset in assets:
            initial_weights.append(ORIGINAL_STRATEGY_ETFS[name][asset])
            # Set bounds based on the strategy's intent
            if name == "Optimistic Outlook (Pro-Risk)":
                if asset == 'Equities': bounds.append((0.70, 0.90)) # High equity
                elif asset == 'Cash': bounds.append((0.0, 0.10)) # Low cash
                else: bounds.append((0.0, 0.20))
            elif name == "Cautiously Optimistic (Balanced)":
                if asset == 'Equities': bounds.append((0.40, 0.60))
                else: bounds.append((0.0, 0.20))
            elif name == "Neutral / Cautious (Defensive)":
                if asset == 'Equities': bounds.append((0.20, 0.40))
                elif asset == 'Bonds': bounds.append((0.30, 0.60)) # Higher bond
                else: bounds.append((0.0, 0.20))
            elif name == "Pessimistic Outlook (Risk-Off)":
                if asset == 'Equities': bounds.append((0.10, 0.30)) # Low equity
                elif asset == 'Bonds' or asset == 'Gold' or asset == 'Cash': bounds.append((0.10, 0.70)) # High safe assets
                else: bounds.append((0.0, 0.10))

    cagr_con = {'type': 'ineq', 'fun': cagr_constraint, 'args': (rebalancing_period, strategy_asset_map)}
    constraints = [cagr_con]
    start_index = 0
    for name in STRATEGY_NAMES:
        num_assets = len(strategy_asset_map[name])
        constraints.append({'type': 'eq', 'fun': lambda w, s=start_index, n=num_assets: 1.0 - np.sum(w[s:s+n])})
        start_index += num_assets

    result = minimize(
        objective_function, initial_weights, args=(rebalancing_period, strategy_asset_map),
        method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 150}
    )
    
    optimized_weights = construct_strategy_weights(result.x, strategy_asset_map)
    return optimized_weights, -result.fun, result.success

def calculate_performance_metrics(portfolio):
    if portfolio.empty: return {"Final Value": 0, "Total Return": 0, "CAGR": 0, "Sharpe Ratio": 0}
    final_value = portfolio['Value'].iloc[-1]
    total_return = (final_value / INITIAL_INVESTMENT) - 1
    cagr = calculate_cagr(portfolio)
    daily_returns = portfolio['Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    return {"Final Value": final_value, "Total Return": total_return, "CAGR": cagr, "Sharpe Ratio": sharpe_ratio}

def plot_results(strategy_df, benchmark_df, original_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(strategy_df.index, strategy_df['Value'], label='Optimized Strategy', color='royalblue', linewidth=2.5)
    ax.plot(original_df.index, original_df['Value'], label='Original Strategy', color='orchid', linewidth=1.5, linestyle=':')
    ax.plot(benchmark_df.index, benchmark_df['Value'], label='S&P 500 Benchmark (SPY)', color='grey', linestyle='--')
    ax.set_title('Optimized Strategy vs. Benchmarks (10-Year Performance)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.figtext(0.5, 0.01, 'Based on a $10,000 initial investment.', ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    if not FRED_API_KEY:
        print("🛑 ERROR: FRED_API_KEY not found in .env file. Please add it.")
    else:
        fetch_all_data()
        
        if ECON_DATA is not None and ETF_RETURNS is not None:
            benchmark_returns = ETF_RETURNS[(ETF_RETURNS.index >= START_DATE) & (ETF_RETURNS.index <= END_DATE)][BENCHMARK_TICKER].dropna()
            benchmark_series = (1 + benchmark_returns).cumprod() * INITIAL_INVESTMENT
            benchmark_portfolio = pd.DataFrame({'Value': benchmark_series})
            benchmark_cagr = calculate_cagr(benchmark_portfolio)
            TARGET_CAGR = benchmark_cagr + CAGR_TARGET_PREMIUM
            print(f"\n--- Benchmark CAGR: {benchmark_cagr:.2%}. Target CAGR: {TARGET_CAGR:.2%} ---")

            all_results = []
            for period in REBALANCING_PERIODS_TO_TEST:
                optimized_weights, sharpe_ratio, success = optimize_strategy(period)
                if success:
                    all_results.append({"period": period, "weights": optimized_weights, "sharpe": sharpe_ratio})
            
            if not all_results:
                print("\n🛑 OPTIMIZATION FAILED: Could not find a strategy that meets the target CAGR. The market may have been too strong.")
                print("Consider lowering the CAGR_TARGET_PREMIUM or adjusting indicator logic.")
            else:
                best_result = max(all_results, key=lambda x: x['sharpe'])
                
                print("\n--- Optimization Complete ---")
                print(f"Best Overall Strategy: {best_result['period']}-Month Rebalancing with a Sharpe Ratio of {best_result['sharpe']:.3f}")
                print("Optimized Weights:")
                for name, assets in best_result['weights'].items():
                    print(f"  {name}:")
                    for asset, weight in assets.items():
                        print(f"    - {asset}: {weight:.2%}")

                best_strategy_portfolio = run_backtest_with_weights(best_result['weights'], best_result['period'])
                original_strategy_portfolio = run_backtest_with_weights(ORIGINAL_STRATEGY_ETFS, 1)
                
                best_metrics = calculate_performance_metrics(best_strategy_portfolio)
                original_metrics = calculate_performance_metrics(original_strategy_portfolio)
                benchmark_metrics = calculate_performance_metrics(benchmark_portfolio)
                
                print("\n--- Final Performance Comparison ---")
                print("-" * 75)
                print(f"| Metric         | {'Optimized Strategy':<20} | {'Original Strategy':<20} | {'S&P 500 Benchmark':<20} |")
                print("-" * 75)
                print(f"| Final Value    | ${best_metrics['Final Value']:<19,.2f} | ${original_metrics['Final Value']:<19,.2f} | ${benchmark_metrics['Final Value']:<19,.2f} |")
                print(f"| Total Return   | {best_metrics['Total Return']:<19.2%} | {original_metrics['Total Return']:<19.2%} | {benchmark_metrics['Total Return']:<19.2%} |")
                print(f"| CAGR           | {best_metrics['CAGR']:<19.2%} | {original_metrics['CAGR']:<19.2%} | {benchmark_metrics['CAGR']:<19.2%} |")
                print(f"| Sharpe Ratio   | {best_metrics['Sharpe Ratio']:<19.3f} | {original_metrics['Sharpe Ratio']:<19.3f} | {benchmark_metrics['Sharpe Ratio']:<19.3f} |")
                print("-" * 75)
                
                plot_results(best_strategy_portfolio, benchmark_portfolio, original_strategy_portfolio)