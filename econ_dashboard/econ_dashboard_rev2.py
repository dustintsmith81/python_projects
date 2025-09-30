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
import re
import requests
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- One-time setup for NLTK's VADER ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')
    print("Download complete.")

# --- API Keys are loaded from your .env file ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuration ---
REFRESH_INTERVAL_MINUTES = 10
NEWS_HEADLINE_COUNT = 50
DATA_START_DATE = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

FINANCIAL_KEYWORDS = {
    'fed', 'federal reserve', 'powell', 'trump', 'interest rate', 'inflation', 'cpi', 'ppi', 'gdp',
    'unemployment', 'jobs report', 'recession', 'growth', 'economy',
    'stocks', 'bonds', 'equities', 'dow jones', 's&p 500', 'nasdaq', 'yields', 'treasury',
    'earnings', 'forecast', 'guidance', 'profit', 'revenue',
    'markets', 'volatile', 'rally', 'correction', 'crash', 'bull market', 'bear market',
    'oil price', 'gold', 'commodities', 'dollar', 'crypto', 'bitcoin'
}
KEYWORD_REGEX = re.compile(r'\b(' + '|'.join(FINANCIAL_KEYWORDS) + r')\b', re.IGNORECASE)

# --- MODIFIED: Switched to a modern dark theme ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# --- NEW: Add custom CSS for a sleeker look ---
app.head = [
    html.Link(rel="preconnect", href="https://fonts.googleapis.com"),
    html.Link(rel="preconnect", href="https://fonts.gstatic.com", crossOrigin="anonymous"),
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap", rel="stylesheet"),
    html.Div(style="""
        body {
            font-family: 'Inter', sans-serif;
        }
        .card {
            background-color: rgba(40, 40, 40, 0.7) !important;
            border: 1px solid #444;
            transition: box-shadow .3s, transform .3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 15px rgba(0, 191, 255, 0.5); /* DeepSkyBlue glow */
        }
        .card-header {
            background-color: rgba(50, 50, 50, 0.8) !important;
            font-weight: bold;
        }
        .display-4 { /* Main Title */
             color: #0dcaf0; /* Cyan */
             text-shadow: 2px 2px 4px #000;
        }
        /* Custom scrollbar for news list */
        .news-list::-webkit-scrollbar {
            width: 8px;
        }
        .news-list::-webkit-scrollbar-track {
            background: #222;
        }
        .news-list::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        .news-list::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
    """)
]

CACHE_CONFIG = {'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': REFRESH_INTERVAL_MINUTES * 60}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


# --- Data Fetching & Analysis Functions (Logic is unchanged) ---
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
            if df.empty:
                error_msg = "Received no data from FRED."
            else:
                print(f"Successfully fetched '{name}' from FRED.")
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
        sp500 = yf.download('^GSPC', period='5y', interval='1d')
        if sp500.empty: return pd.DataFrame(), "yfinance returned no data."
        return sp500, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance Error: {e}"

@cache.memoize()
def get_top_headlines():
    print("--- FETCHING & FILTERING LIVE DATA: News Headlines (NewsAPI) ---")
    if not NEWS_API_KEY or NEWS_API_KEY == "PASTE_YOUR_NEWS_API_KEY_HERE":
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

@cache.memoize()
def get_llm_investment_ideas(recommendation):
    """
    Calls the Gemini API using the official Python SDK to get specific investment ideas.
    """
    print("--- QUERYING LLM: For Investment Ideas (Direct API Call) ---")
    if GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return {"Error": "Gemini API Key not configured."}
    
    try:
        client = genai.Client(api_key = GEMINI_API_KEY)

        prompt = f"""
            Based on the following investment strategy, provide a diversified list of specific, real-world examples of publicly traded investment instruments for a moderate-risk portfolio. For each category, provide 2-3 well-known examples with their stock tickers.
            **Investment Strategy:** {recommendation['title']}
            **Rationale:** {recommendation['details']}
            **Example Allocation:** {recommendation['portfolio']}
            Format your response as a valid JSON object with keys matching the asset classes in the example allocation (e.g., "Equities", "Bonds"). The value for each key should be a list of strings, where each string is formatted as "Name (TICKER)".
            """
        
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                system_instruction="You are a Wall Street financial analyst and trader"
                )
            )
        
        json_content = response.text.strip().replace('```json', '').replace('```', '')
        ideas = json.loads(json_content)
        print("Successfully received and parsed investment ideas from LLM.")
        return ideas

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        error_text = http_err.response.text
        print(f"ERROR: LLM API HTTP Error {status_code}: {error_text}")
        if status_code == 400:
            return {"Error": "AI Error (400): Bad Request. Check Gemini Key or prompt format."}
        elif status_code == 403:
            return {"Error": "AI Error (403): Forbidden. Please ENABLE the 'Generative Language API' in your Google Cloud project and ensure billing is linked."}
        elif status_code == 404:
            return {"Error": "AI Error (404): Not Found. The specific model endpoint is incorrect."}
        else:
            return {"Error": f"AI model connection failed. Status: {status_code}."}
    except requests.exceptions.RequestException as e:
        print(f"ERROR: LLM API call failed: {e}")
        return {"Error": "Could not connect to the AI model. Check network connection."}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to parse LLM response: {e}")
        return {"Error": "Received an invalid response from the AI model."}

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
    if not headlines or not isinstance(headlines, list) or headlines[0]['url'] == '#':
        return "N/A", "secondary", [], 0
    analyzer = SentimentIntensityAnalyzer()
    headlines_with_scores, compound_scores = [], []
    for h in headlines:
        score = analyzer.polarity_scores(h['title'])['compound']
        compound_scores.append(score)
        if score >= 0.05: color = "success"
        elif score <= -0.05: color = "danger"
        else: color = "secondary"
        h['score'], h['color'] = score, color
        headlines_with_scores.append(h)
    average_score = sum(compound_scores) / len(compound_scores)
    if average_score >= 0.05: avg_sentiment, avg_color = "Positive", "success"
    elif average_score <= -0.05: avg_sentiment, avg_color = "Negative", "danger"
    else: avg_sentiment, avg_color = "Neutral", "info"
    return f"{avg_sentiment} ({average_score:.2f})", avg_color, headlines_with_scores


def interpret_indicator(indicator_name, series, is_good_high, error_msg):
    if error_msg: return f"❗️ Data Fetch Error: {error_msg}"
    series = series.dropna()
    if series.empty or len(series) < 2: return "ℹ️ Not enough data."
    latest_value, quarterly_avg, yearly_avg = series.iloc[-1], series.last('3M').mean(), series.last('1Y').mean()
    status = "near"
    if latest_value > quarterly_avg * 1.02: status = "trending strongly above"
    elif latest_value > yearly_avg: status = "above"
    elif latest_value < quarterly_avg * 0.98: status = "trending strongly below"
    elif latest_value < yearly_avg: status = "below"
    baseline_text = f"its quarterly average ({quarterly_avg:.2f})."
    if (is_good_high and latest_value > yearly_avg) or (not is_good_high and latest_value < yearly_avg):
        interpretation = f"✅ Positive: Latest ({latest_value:.2f}) is {status} {baseline_text}"
    else:
        interpretation = f"⚠️ Caution: Latest ({latest_value:.2f}) is {status} {baseline_text}"
    return interpretation


def generate_investment_recommendations(indicators, spreads, market_data):
    optimized_weights = {
        "Optimistic Outlook (Pro-Risk)": {"Equities": 0.70, "Bonds": 0.10, "Commodities": 0.20, "Cash": 0.00},
        "Cautiously Optimistic (Balanced)": {"Equities": 0.60, "Bonds": 0.20, "Real Estate": 0.10, "Cash": 0.10},
        "Neutral / Cautious (Defensive)": {"Equities": 0.20, "Bonds": 0.40, "Alternatives": 0.20, "Cash": 0.20},
        "Pessimistic Outlook (Risk-Off)": {"Equities": 0.10, "Bonds": 0.10, "Gold": 0.70, "Cash": 0.10}
    }
    market_perf, market_error = market_data
    score = 0
    all_metrics = {**indicators, **spreads}
    for name, (df, is_good_high, err) in all_metrics.items():
        if err or df.empty: continue
        series = df.iloc[:, 0].dropna()
        if not series.empty:
            latest, mean = series.iloc[-1], series.mean()
            weight = 2 if name in spreads else 1
            if (is_good_high and latest > mean) or (not is_good_high and latest < mean): score += weight
            else: score -= weight
    if not market_error and not market_perf.empty and 'Close' in market_perf.columns:
        market_perf['MA50'] = market_perf['Close'].rolling(window=50).mean()
        market_perf['MA200'] = market_perf['Close'].rolling(window=200).mean()
        if not market_perf['MA50'].dropna().empty and not market_perf['MA200'].dropna().empty:
            if market_perf['MA50'].iloc[-1] > market_perf['MA200'].iloc[-1]: score += 3
            else: score -= 2

    if score >= 8: strategy_name = "Optimistic Outlook (Pro-Risk)"
    elif score >= 2: strategy_name = "Cautiously Optimistic (Balanced)"
    elif score > -5: strategy_name = "Neutral / Cautious (Defensive)"
    else: strategy_name = "Pessimistic Outlook (Risk-Off)"
    
    details_map = {
        "Optimistic Outlook (Pro-Risk)": "Economic indicators are broadly positive, credit conditions appear favorable, and market momentum is strong. Consider overweighting equities and growth-oriented assets.",
        "Cautiously Optimistic (Balanced)": "A mix of positive and negative signals suggests a balanced approach. The economy shows resilience but some headwinds exist. Maintain a diversified, moderate-risk portfolio.",
        "Neutral / Cautious (Defensive)": "Economic data is mixed to negative, with potential risks from credit markets. A defensive posture is warranted. Focus on quality and capital preservation.",
        "Pessimistic Outlook (Risk-Off)": "Most indicators show warning signs, credit spreads may be widening, and market trends are negative. Prioritize capital preservation and reduce exposure to volatile assets."
    }
    portfolio_dict = optimized_weights[strategy_name]
    portfolio_string = ", ".join([f"{weight * 100:.0f}% {asset}" for asset, weight in portfolio_dict.items()])
    return {"title": strategy_name, "details": details_map[strategy_name], "portfolio": portfolio_string}


# --- UI and Graphing Functions ---
def create_gauge_card(score):
    score = round(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text': "6-Month Momentum", 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'color': 'white'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "rgba(0,0,0,0)"},
               'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#888",
               'steps': [{'range': [0, 35], 'color': '#dc3545'}, {'range': [35, 65], 'color': '#0dcaf0'},
                         {'range': [65, 100], 'color': '#198754'}],
               'threshold': {'line': {'color': "white", 'width': 5}, 'thickness': 0.9, 'value': score}}))
    fig.update_layout(height=300, margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return dbc.Card(dbc.CardBody([
        # --- MODIFIED: Changed text-light to text-info for more pop ---
        html.H4("Forward-Looking Economic Score", className="card-title text-center text-info"),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ]), className="mb-4")


def create_indicator_graph(df, name, error_msg):
    fig = go.Figure()
    fig.update_layout(title={'text': name, 'x': 0.5, 'font': {'size': 16, 'color': 'white'}},
                      margin=dict(l=20, r=20, t=40, b=20), height=250,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(gridcolor='#444', color='white'), yaxis=dict(gridcolor='#444', color='white'),
                      legend=dict(font=dict(color='white')))
    if error_msg:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[
            {"text": error_msg, "xref": "paper", "yref": "paper", "showarrow": False,
             "font": {"size": 12, "color": "red"}}])
        return fig
    if df.empty or df.columns[0] not in df or df[df.columns[0]].dropna().empty:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[
            {"text": "No Data Available", "xref": "paper", "yref": "paper", "showarrow": False,
             "font": {"size": 12}}])
        return fig
    df_monthly = df.resample('MS').mean()
    df_last_6m = df_monthly.last('6M')
    fig.add_trace(go.Scatter(x=df_last_6m.index, y=df_last_6m[df.columns[0]], mode='lines+markers',
                            name='6-Month Trend', line=dict(color='#0dcaf0')))
    yearly_avg = df_monthly.last('1Y')[df.columns[0]].mean()
    fig.add_hline(y=yearly_avg, line_dash="dot", annotation_text="1-Year Avg",
                  annotation_position="bottom right", line_color='grey', annotation_font_color='grey')
    return fig


app.layout = dbc.Container(fluid=True, className="p-4", children=[
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MINUTES * 60 * 1000, n_intervals=0),
    dcc.Loading(id="loading-spinner", type="default", children=[html.Div(id='main-dashboard-content')])
])


@app.callback(Output('main-dashboard-content', 'children'), Input('interval-component', 'n_intervals'))
def update_dashboard_content(n):
    print(f"\n--- Building Dashboard Layout (Interval: {n}) ---")
    indicators, spreads, market_perf_data, headlines = get_economic_indicators(), get_credit_spreads(), get_market_performance(), get_top_headlines()
    recommendation = generate_investment_recommendations(indicators, spreads, market_perf_data)
    llm_ideas = get_llm_investment_ideas(recommendation)
    economic_score = calculate_economic_score(indicators, spreads)
    sentiment_text, sentiment_color, headlines_with_scores = analyze_headline_sentiment(headlines)
    num_analyzed = len(headlines_with_scores)
    score_card = create_gauge_card(economic_score)
    llm_idea_components = []
    if "Error" in llm_ideas:
        llm_idea_components = [dbc.Alert(llm_ideas["Error"], color="danger", style={'white-space': 'pre-wrap'})]
    else:
        for category, examples in llm_ideas.items():
            llm_idea_components.append(html.H6(f"{category}:", className="text-info"))
            llm_idea_components.append(html.Ul([html.Li(ex) for ex in examples]))

    recommendation_card = dbc.Card([
        dbc.CardHeader("Investment Strategy"),
        dbc.CardBody([
            html.H5(recommendation['title'], className="card-title text-primary"),
            html.P(recommendation['details']), html.Hr(),
            html.P("Optimized Allocation:", className="font-weight-bold"),
            html.P(recommendation['portfolio']), html.Hr(),
            html.H5("AI-Generated Investment Ideas", className="text-primary mt-4"),
            *llm_idea_components,
            html.Small(
                "Disclaimer: AI-generated examples are for informational purposes only and are NOT financial advice. Always conduct your own research.",
                className="text-danger mt-3 fst-italic")
        ])
    ], className="mb-4 h-100")

    news_card = dbc.Card([
        dbc.CardHeader("Market-Moving Headlines"),
        dbc.CardBody([
            html.Div([html.H6("Overall News Sentiment:", className="d-inline-block me-2"),
                      html.H5(sentiment_text, className=f"d-inline-block text-{sentiment_color}")],
                     className="text-center"),
            html.P(f"(Analyzed from {num_analyzed} relevant headlines)", className="text-muted small text-center"),
            html.Hr(),
            html.Ul(
                [html.Li([dbc.Badge(f"{h['score']:.2f}", color=h['color'], className="me-2 flex-shrink-0"),
                          html.A(h['title'], href=h['url'], target='_blank', className="flex-grow-1")],
                         className="d-flex align-items-center mb-2") for h in headlines_with_scores],
                className="list-unstyled news-list",
                style={'maxHeight': '350px', 'overflowY': 'auto', 'fontSize': '0.9rem'})
        ])
    ], className="mb-4 h-100")

    indicator_cards = [dbc.Col(dbc.Card([dcc.Graph(figure=create_indicator_graph(df, df.columns[0], error_msg),
                                                 config={'displayModeBar': False}),
                                        dbc.CardBody(html.P(
                                            interpret_indicator(df.columns[0], df[df.columns[0]], is_good_high,
                                                                error_msg), className="card-text small text-center"),
                                                     className="p-2")], className="mb-4"), xl=3, lg=4, md=6,
                               sm=12) for name, (df, is_good_high, error_msg) in indicators.items()]
    spread_cards = [dbc.Col(dbc.Card([dcc.Graph(figure=create_indicator_graph(df, df.columns[0], error_msg),
                                                config={'displayModeBar': False}),
                                       dbc.CardBody(html.P(
                                           interpret_indicator(df.columns[0], df[df.columns[0]], is_good_high,
                                                               error_msg), className="text-muted small text-center"),
                                                    className="p-2")], className="mb-4"), lg=6, md=6,
                              sm=12) for name, (df, is_good_high, error_msg) in spreads.items()]

    print("--- Layout Update Complete ---")
    return html.Div([
        dbc.Row(
            [dbc.Col(html.H1("Live Economic & Market Dashboard", className="display-4 text-center my-4"), width=12)]),
        dbc.Row([dbc.Col(score_card, lg=8, md=10)], justify="center", className="mb-4"),
        dbc.Row([dbc.Col(recommendation_card, lg=6), dbc.Col(news_card, lg=6)], className="mb-4"),
        html.H3("Key Economic Indicators", className="mt-4"), html.Hr(),
        dbc.Row(indicator_cards),
        html.H3("Key Credit Spreads", className="mt-4"), html.Hr(),
        dbc.Row(spread_cards)
    ])


if __name__ == '__main__':
    if any(k is None or k == "" for k in [FRED_API_KEY, NEWS_API_KEY, GEMINI_API_KEY]):
        print("\n" + "=" * 80)
        print("🛑 WARNING: ONE OR MORE API KEYS ARE MISSING!")
        print("Please check your .env file and ensure all API keys are set.")
        print("=" * 80 + "\n")
    app.run(debug=True, use_reloader=False)