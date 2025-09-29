import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader.data as web
from newsapi import NewsApiClient
import os
from flask_caching import Cache
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re # Used for keyword searching

load_dotenv()

# --- One-time setup for NLTK's VADER ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')
    print("Download complete.")

# --- ⬇️ ACTION REQUIRED: PASTE YOUR API KEYS HERE ⬇️ ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "PASTE_YOUR_FRED_API_KEY_HERE")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "PASTE_YOUR_NEWS_API_KEY_HERE")
# --- ⬆️ ACTION REQUIRED: END ⬆️ ---

# --- Configuration ---
REFRESH_INTERVAL_MINUTES = 10
NEWS_HEADLINE_COUNT = 50 
DATA_START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

FINANCIAL_KEYWORDS = {
    'fed', 'federal reserve', 'powell', 'trump', 'interest rate', 'inflation', 'cpi', 'ppi', 'gdp', 
    'unemployment', 'jobs report', 'recession', 'growth', 'economy',
    'stocks', 'bonds', 'equities', 'dow jones', 's&p 500', 'nasdaq', 'yields', 'treasury',
    'earnings', 'forecast', 'guidance', 'profit', 'revenue',
    'markets', 'volatile', 'rally', 'correction', 'crash', 'bull market', 'bear market',
    'oil price', 'gold', 'commodities', 'dollar', 'crypto', 'bitcoin'
}
KEYWORD_REGEX = re.compile(r'\b(' + '|'.join(FINANCIAL_KEYWORDS) + r')\b', re.IGNORECASE)


# --- Dash App & Cache Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
CACHE_CONFIG = {'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': REFRESH_INTERVAL_MINUTES * 60}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

# --- Data Fetching Functions ---
@cache.memoize()
def get_economic_indicators():
    print("--- FETCHING LIVE DATA: Economic Indicators (FRED) ---")
    indicator_series = {
        'Industrial Production Index': ('fred-datareader', 'INDPRO', True),
        'Chicago Fed National Activity Index': ('fred-datareader', 'CFNAI', True),
        'Consumer Sentiment Index': ('fred-datareader', 'UMCSENT', True),
        'New Privately-Owned Housing Units Started': ('fred-datareader', 'HOUST', True),
        'Initial Jobless Claims': ('fred-datareader', 'ICSA', False),
        'Durable Goods Orders (MoM %)': ('fred-datareader', 'DGORDER', True),
        'Retail Sales (MoM %)': ('fred-datareader', 'RSXFS', True),
        'Building Permits (Thousands)': ('fred-datareader', 'PERMIT', True),
        'Unemployment Rate (%)': ('fred-datareader', 'UNRATE', False),
        'Average Weekly Hours, Manufacturing': ('fred-datareader', 'AWHMAN', True),
    }
    indicators = {}
    for name, (source, series_id, is_good_high) in indicator_series.items():
        df, error_msg = pd.DataFrame(), None
        try:
            df = web.DataReader(series_id, 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
            df.rename(columns={series_id: name}, inplace=True)
            if df.empty: error_msg = "Received no data from FRED."
            else: print(f"Successfully fetched '{name}' from FRED.")
        except Exception as e:
            error_msg = f"Error: Check FRED Key or Network."
            print(f"ERROR fetching '{name}': {e}")
        if error_msg: 
            df = pd.DataFrame({name: []}, index=pd.to_datetime([]))
        indicators[name] = (df, is_good_high, error_msg)
    return indicators

@cache.memoize()
def get_credit_spreads():
    print("--- FETCHING LIVE DATA: Credit Spreads (FRED) ---")
    spreads = {}
    try:
        hy_spread_df = web.DataReader('BAMLH0A0HYM2', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        hy_spread_df.rename(columns={'BAMLH0A0HYM2': 'High-Yield Spread (OAS)'}, inplace=True)
        spreads['High-Yield Spread (OAS)'] = (hy_spread_df, False, None)
        print("Successfully fetched 'High-Yield Spread (OAS)'.")
    except Exception as e:
        print(f"ERROR fetching High-Yield Spread: {e}")
        df = pd.DataFrame({'High-Yield Spread (OAS)': []}, index=pd.to_datetime([]))
        spreads['High-Yield Spread (OAS)'] = (df, False, "FRED Fetch Failed")
    try:
        baa_yield = web.DataReader('DBAA', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        ten_year_yield = web.DataReader('DGS10', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        combined = baa_yield.join(ten_year_yield, how='inner')
        combined['Baa - 10-Year Treasury'] = combined['DBAA'] - combined['DGS10']
        spreads['Baa - 10-Year Treasury'] = (combined[['Baa - 10-Year Treasury']], False, None)
        print("Successfully calculated 'Baa - 10-Year Treasury'.")
    except Exception as e:
        print(f"ERROR fetching Baa/10-Year components: {e}")
        df = pd.DataFrame({'Baa - 10-Year Treasury': []}, index=pd.to_datetime([]))
        spreads['Baa - 10-Year Treasury'] = (df, False, "FRED Fetch Failed")
    return spreads

@cache.memoize()
def get_market_performance():
    print("--- FETCHING LIVE DATA: Market Performance (yfinance) ---")
    try:
        sp500 = yf.download('^GSPC', period='5y', interval='1d', auto_adjust=True)
        if sp500.empty: return pd.DataFrame(), "yfinance returned no data."
        return sp500, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance Error: {e}"

@cache.memoize()
def get_top_headlines():
    print("--- FETCHING & FILTERING LIVE DATA: News Headlines (NewsAPI) ---")
    if NEWS_API_KEY == "PASTE_YOUR_NEWS_API_KEY_HERE":
        return [{'title': "NewsAPI key not configured.", 'url': '#'}]
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        top_headlines = newsapi.get_top_headlines(category='business', language='en', country='us', page_size=100)
        articles = top_headlines.get('articles', [])
        if not articles:
            return [{'title': "No headlines returned from NewsAPI.", 'url': '#'}]
        relevant_headlines = []
        for article in articles:
            title = article.get('title')
            url = article.get('url')
            if title and url and KEYWORD_REGEX.search(title):
                relevant_headlines.append({'title': title, 'url': url})
        print(f"Found {len(relevant_headlines)} relevant headlines out of {len(articles)}.")
        if not relevant_headlines:
            return [{'title': "No financially relevant headlines found.", 'url': '#'}]
        return relevant_headlines[:NEWS_HEADLINE_COUNT]
    except Exception as e:
        error_message = f"Failed to fetch news: {e}"
        if 'apiKeyInvalid' in str(e): error_message = "NewsAPI Key is invalid."
        if 'rateLimited' in str(e): error_message = "NewsAPI rate limit exceeded."
        return [{'title': error_message, 'url': '#'}]

# --- Calculation & Analysis Functions ---

def calculate_economic_score(indicators, spreads):
    score, max_score = 0, 0
    all_metrics = {**indicators, **spreads}
    for name, (df, is_good_high, err) in all_metrics.items():
        if err or df.empty: continue
        series = df.iloc[:, 0].dropna()
        if len(series) > 30:
            latest_value = series.iloc[-1]
            six_month_avg = series.rolling(window=180, min_periods=30).mean().iloc[-1]
            if pd.isna(six_month_avg): continue
            weight = 2 if name in spreads else 1
            max_score += weight
            is_trending_positive = latest_value > six_month_avg
            if (is_trending_positive and is_good_high) or (not is_trending_positive and not is_good_high):
                score += weight
    if max_score == 0: return 50
    return (score / max_score) * 100

def analyze_headline_sentiment(headlines):
    """
    Analyzes sentiment, returns the average score, and the list of headlines
    with their individual scores and colors.
    """
    if not headlines or not isinstance(headlines, list) or headlines[0]['url'] == '#':
        return "N/A", "secondary", [], 0

    analyzer = SentimentIntensityAnalyzer()
    headlines_with_scores = []
    compound_scores = []

    for h in headlines:
        score = analyzer.polarity_scores(h['title'])['compound']
        compound_scores.append(score)
        
        if score >= 0.05: color = "success"  # Green for positive
        elif score <= -0.05: color = "danger"   # Red for negative
        else: color = "secondary"  # Gray for neutral
            
        h['score'] = score
        h['color'] = color
        headlines_with_scores.append(h)

    average_score = sum(compound_scores) / len(compound_scores)
    
    if average_score >= 0.05: avg_sentiment, avg_color = "Positive", "success"
    elif average_score <= -0.05: avg_sentiment, avg_color = "Negative", "danger"
    else: avg_sentiment, avg_color = "Neutral", "info"
        
    return f"{avg_sentiment} ({average_score:.2f})", avg_color, headlines_with_scores

def interpret_indicator(indicator_name, series, is_good_high, error_msg):
    if error_msg: return f"❗️ Data Fetch Error: {error_msg}"
    series = series.dropna()
    if series.empty or len(series) < 2: return "ℹ️ Not enough data for interpretation."
    latest_value = series.iloc[-1]
    quarterly_avg = series.last('3M').mean()
    yearly_avg = series.last('1Y').mean()
    status = "near"
    if latest_value > quarterly_avg * 1.02: status = "trending strongly above"
    elif latest_value > yearly_avg: status = "above"
    elif latest_value < quarterly_avg * 0.98: status = "trending strongly below"
    elif latest_value < yearly_avg: status = "below"
    baseline_text = f"its quarterly average ({quarterly_avg:.2f})."
    if (is_good_high and latest_value > yearly_avg) or (not is_good_high and latest_value < yearly_avg):
        interpretation = f"✅ Positive: Latest ({latest_value:.2f}) is {status} {baseline_text}"
    else: interpretation = f"⚠️ Caution: Latest ({latest_value:.2f}) is {status} {baseline_text}"
    return interpretation

def generate_investment_recommendations(indicators, spreads, market_data):
    # This function remains unchanged
    market_perf, market_error = market_data
    score = 0
    all_metrics = {**indicators, **spreads}
    for name, (df, is_good_high, err) in all_metrics.items():
        if err or df.empty: continue
        series = df.iloc[:, 0].dropna()
        if not series.empty:
            latest = series.iloc[-1]; mean = series.mean()
            weight = 2 if name in spreads else 1
            if (is_good_high and latest > mean) or (not is_good_high and latest < mean): score += weight
            else: score -= weight
    if not market_error and not market_perf.empty and 'Close' in market_perf.columns:
        market_perf['MA50'] = market_perf['Close'].rolling(window=50).mean()
        market_perf['MA200'] = market_perf['Close'].rolling(window=200).mean()
        if not market_perf['MA50'].dropna().empty and not market_perf['MA200'].dropna().empty:
            if market_perf['MA50'].iloc[-1] > market_perf['MA200'].iloc[-1]: score += 3
            else: score -= 2
    if score >= 8: return {"title": "Optimistic Outlook (Pro-Risk)", "details": "Economic indicators are broadly positive, credit conditions appear favorable, and market momentum is strong. Consider overweighting equities and growth-oriented assets.", "portfolio": "65% Equities (e.g., VTI, QQQ), 20% Bonds (e.g., BND), 10% Commodities (e.g., GLD), 5% Cash."}
    elif score >= 2: return {"title": "Cautiously Optimistic (Balanced)", "details": "A mix of positive and negative signals suggests a balanced approach. The economy shows resilience but some headwinds exist. Maintain a diversified, moderate-risk portfolio.", "portfolio": "55% Equities (e.g., VTI, VEA), 30% Bonds (e.g., AGG, TIP), 10% Real Estate (e.g., VNQ), 5% Cash."}
    elif score > -5: return {"title": "Neutral / Cautious (Defensive)", "details": "Economic data is mixed to negative, with potential risks from credit markets. A defensive posture is warranted. Focus on quality and capital preservation.", "portfolio": "40% Equities (e.g., SPLV, USMV), 45% Bonds (e.g., IEF, TLT), 10% Alternatives (e.g., IAU), 5% Cash."}
    else: return {"title": "Pessimistic Outlook (Risk-Off)", "details": "Most indicators show warning signs, credit spreads may be widening, and market trends are negative. Prioritize capital preservation and reduce exposure to volatile assets.", "portfolio": "25% Equities (e.g., VOO, defensive sectors), 55% Government Bonds (e.g., GOVT), 15% Gold (e.g., GLD), 5% Cash."}

# --- UI and Graphing Functions ---

def create_gauge_card(score):
    score = round(score)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = score,
        title = {'text': "6-Month Momentum", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0.4)"},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#dc3545'},
                {'range': [35, 65], 'color': '#0dcaf0'},
                {'range': [65, 100], 'color': '#198754'}],
            'threshold': {'line': {'color': "black", 'width': 5}, 'thickness': 0.9, 'value': score}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white")
    return dbc.Card(dbc.CardBody([
            html.H4("Forward-Looking Economic Score", className="card-title text-center"),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P("This score measures indicator momentum against their 6-month average.", className="text-muted small text-center mt-2")
        ]), className="shadow-sm mb-4")

def create_indicator_graph(df, name, error_msg):
    fig = go.Figure()
    fig.update_layout(title={'text': name, 'x': 0.5, 'font': {'size': 16}}, margin=dict(l=20, r=20, t=40, b=20), height=250, plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9')
    if error_msg:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": error_msg, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 12, "color": "red"}}])
        return fig
    if df.empty or df.columns[0] not in df or df[df.columns[0]].dropna().empty: 
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": "No Data Available", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 12}}])
        return fig
    df_monthly = df.resample('MS').mean()
    df_last_6m = df_monthly.last('6M')
    fig.add_trace(go.Scatter(x=df_last_6m.index, y=df_last_6m[df.columns[0]], mode='lines+markers', name='6-Month Trend'))
    yearly_avg = df_monthly.last('1Y')[df.columns[0]].mean()
    fig.add_hline(y=yearly_avg, line_dash="dot", annotation_text="1-Year Avg", annotation_position="bottom right", line_color='grey')
    return fig

app.layout = dbc.Container(fluid=True, className="p-4", style={'backgroundColor': '#f0f2f5'}, children=[
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MINUTES * 60 * 1000, n_intervals=0),
    dcc.Loading(id="loading-spinner", type="default", children=[html.Div(id='dashboard-content')])
])

@app.callback(Output('dashboard-content', 'children'), Input('interval-component', 'n_intervals'))
def update_dashboard_content(n):
    print(f"\n--- Building Dashboard Layout (Interval: {n}) ---")
    indicators = get_economic_indicators()
    spreads = get_credit_spreads()
    market_perf_data = get_market_performance()
    headlines = get_top_headlines()
    
    recommendation = generate_investment_recommendations(indicators, spreads, market_perf_data)
    economic_score = calculate_economic_score(indicators, spreads)
    sentiment_text, sentiment_color, headlines_with_scores = analyze_headline_sentiment(headlines)
    num_analyzed = len(headlines_with_scores)

    score_card = create_gauge_card(economic_score)
    recommendation_card = dbc.Card([dbc.CardHeader("Investment Strategy Recommendation (Moderate Risk)"), dbc.CardBody([html.H5(recommendation['title'], className="card-title text-info"), html.P(recommendation['details'], className="card-text"), html.Hr(), html.P("Example Portfolio Allocation:", className="font-weight-bold"), html.P(recommendation['portfolio']), html.Small("Disclaimer: This is not financial advice.", className="text-muted")])], className="shadow-sm mb-4")
    
    # --- MODIFIED: News card now renders clickable headlines with individual scores ---
    news_card = dbc.Card([
        dbc.CardHeader("Top Market-Moving Headlines"),
        dbc.CardBody([
            html.Div([
                html.H6("Overall News Sentiment:", className="d-inline-block me-2"),
                html.H5(sentiment_text, className=f"d-inline-block text-{sentiment_color}")
            ], className="text-center"),
            html.P(f"(Analyzed from {num_analyzed} relevant headlines)", className="text-muted small text-center"),
            html.Hr(),
            # Create a list of list items, each containing a score badge and a clickable link
            html.Ul(
                [
                    html.Li(
                        [
                            dbc.Badge(f"{h['score']:.2f}", color=h['color'], className="me-2 flex-shrink-0"),
                            html.A(h['title'], href=h['url'], target='_blank', className="flex-grow-1")
                        ],
                        className="d-flex align-items-center mb-2"
                    ) for h in headlines_with_scores
                ],
                className="list-unstyled",
                style={'maxHeight': '240px', 'overflowY': 'auto', 'fontSize': '0.9rem'}
            )
        ])
    ], className="shadow-sm mb-4")

    indicator_cards = [dbc.Col(dbc.Card([dcc.Graph(figure=create_indicator_graph(df, df.columns[0], error_msg), config={'displayModeBar': False}), dbc.CardBody(html.P(interpret_indicator(df.columns[0], df[df.columns[0]], is_good_high, error_msg), className="card-text small text-center"), className="p-2")], className="mb-4 shadow-sm"), xl=3, lg=4, md=6, sm=12) for name, (df, is_good_high, error_msg) in indicators.items()]
    spread_cards = [dbc.Col(dbc.Card([dcc.Graph(figure=create_indicator_graph(df, df.columns[0], error_msg), config={'displayModeBar': False}), dbc.CardBody(html.P(interpret_indicator(df.columns[0], df[df.columns[0]], is_good_high, error_msg), className="card-text small text-center"), className="p-2")], className="mb-4 shadow-sm"), lg=6, md=6, sm=12) for name, (df, is_good_high, error_msg) in spreads.items()]
    
    print("--- Layout Update Complete ---")
    return html.Div([
        dbc.Row([dbc.Col(html.H1("Live Economic Indicators Dashboard", className="text-center text-primary mb-4"), width=12)]),
        dbc.Row([dbc.Col(score_card, lg=8, md=10)], justify="center"),
        dbc.Row([dbc.Col(recommendation_card, lg=6), dbc.Col(news_card, lg=6)]),
        html.H3("Key Economic Indicators", className="mt-4 text-secondary"),
        html.Hr(),
        dbc.Row(indicator_cards),
        html.H3("Key Credit Spreads", className="mt-5 text-secondary"),
        html.Hr(),
        dbc.Row(spread_cards)
    ])

if __name__ == '__main__':
    if FRED_API_KEY == "PASTE_YOUR_FRED_API_KEY_HERE" or NEWS_API_KEY == "PASTE_YOUR_NEWS_API_KEY_HERE":
        print("\n" + "="*80)
        print("🛑 WARNING: API KEY(S) ARE MISSING!")
        print("Please paste your FRED and NewsAPI keys at the top of the script.")
        print("="*80 + "\n")
    app.run(debug=True, use_reloader=False)

