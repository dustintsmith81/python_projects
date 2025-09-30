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

# --- API Keys ---
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

# --- App Initialization & Styling ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])
server = app.server
app.head = [
    html.Link(rel="preconnect", href="https://fonts.googleapis.com"),
    html.Link(rel="preconnect", href="https://fonts.gstatic.com", crossOrigin="anonymous"),
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap", rel="stylesheet"),
    html.Div(style="""
        body { font-family: 'Inter', sans-serif; background-color: #060606; background-image: radial-gradient(circle at top right, rgba(13, 202, 240, 0.1), transparent 40%), radial-gradient(circle at bottom left, rgba(13, 110, 253, 0.1), transparent 50%); }
        .card { background-color: rgba(30, 30, 30, 0.75) !important; border: 1px solid #444; backdrop-filter: blur(10px); animation: fadeIn 0.5s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .card:hover { transform: translateY(-5px); box-shadow: 0 0 15px rgba(0, 191, 255, 0.5); }
        .card-header { background-color: rgba(45, 45, 45, 0.8) !important; font-weight: bold; }
        .display-4 { color: #0dcaf0; text-shadow: 2px 2px 4px #000; }
        .news-list a { color: #69cde4; text-decoration: none; }
        .news-list a:hover { color: #fff; text-decoration: underline; }
        .news-list::-webkit-scrollbar { width: 8px; }
        .news-list::-webkit-scrollbar-track { background: #222; }
        .news-list::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
        .news-list::-webkit-scrollbar-thumb:hover { background: #777; }
    """)
]
CACHE_CONFIG = {'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': REFRESH_INTERVAL_MINUTES * 60}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


# --- Data Fetching & Analysis ---
@cache.memoize()
def get_economic_indicators():
    print("--- FETCHING: Economic Indicators ---")
    indicator_series = {
        'Industrial Production Index': ('INDPRO', True), 'Chicago Fed National Activity Index': ('CFNAI', True),
        'Consumer Sentiment Index': ('UMCSENT', True), 'New Privately-Owned Housing Units Started': ('HOUST', True),
        'Initial Jobless Claims': ('ICSA', False), 'Durable Goods Orders (MoM %)': ('DGORDER', True),
        'Retail Sales (MoM %)': ('RSXFS', True), 'Building Permits (Thousands)': ('PERMIT', True),
        'Unemployment Rate (%)': ('UNRATE', False), 'Average Weekly Hours, Manufacturing': ('AWHMAN', True),
    }
    indicators = {}
    for name, (series_id, is_good_high) in indicator_series.items():
        df, error_msg = pd.DataFrame(), None
        try:
            df = web.DataReader(series_id, 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
            df.rename(columns={series_id: name}, inplace=True)
            if df.empty: error_msg = "No data from FRED."
        except Exception as e:
            error_msg = f"FRED API Error."
        if error_msg: df = pd.DataFrame({name: []}, index=pd.to_datetime([]))
        indicators[name] = (df, is_good_high, error_msg)
    return indicators


@cache.memoize()
def get_credit_spreads():
    print("--- FETCHING: Credit Spreads ---")
    spreads = {}
    # --- High-Yield Spread ---
    try:
        hy_spread_df = web.DataReader('BAMLH0A0HYM2', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        hy_spread_df.rename(columns={'BAMLH0A0HYM2': 'High-Yield Spread (OAS)'}, inplace=True)
        spreads['High-Yield Spread (OAS)'] = (hy_spread_df, False, None)
    except Exception as e:
        print(f"ERROR fetching High-Yield Spread: {e}")
        # --- FIX: Create a DataFrame with the correct column name on error ---
        df = pd.DataFrame({'High-Yield Spread (OAS)': []}, index=pd.to_datetime([]))
        spreads['High-Yield Spread (OAS)'] = (df, False, "API Error")
    
    # --- Baa - 10-Year Treasury Spread ---
    try:
        baa_yield = web.DataReader('DBAA', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        ten_year_yield = web.DataReader('DGS10', 'fred', DATA_START_DATE, api_key=FRED_API_KEY)
        combined = baa_yield.join(ten_year_yield, how='inner')
        combined['Baa - 10-Year Treasury'] = combined['DBAA'] - combined['DGS10']
        spreads['Baa - 10-Year Treasury'] = (combined[['Baa - 10-Year Treasury']], False, None)
    except Exception as e:
        print(f"ERROR fetching Baa/10-Year components: {e}")
        # --- FIX: Create a DataFrame with the correct column name on error ---
        df = pd.DataFrame({'Baa - 10-Year Treasury': []}, index=pd.to_datetime([]))
        spreads['Baa - 10-Year Treasury'] = (df, False, "API Error")
    return spreads


@cache.memoize()
def get_market_performance():
    print("--- FETCHING: Market Performance & Technicals ---")
    try:
        market_data = yf.download(['^GSPC', '^VIX'], period='1y', interval='1d')
        if market_data.empty: return {}, "yfinance returned no data."
        sp500 = market_data['Close']['^GSPC'].dropna()
        delta = sp500.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        technicals = {
            "sp500_ma50": sp500.rolling(window=50).mean().iloc[-1],
            "sp500_ma200": sp500.rolling(window=200).mean().iloc[-1],
            "sp500_rsi": 100 - (100 / (1 + rs.iloc[-1])),
            "vix": market_data['Close']['^VIX'].dropna().iloc[-1]
        }
        return technicals, None
    except Exception as e:
        return {}, f"yfinance Error: {e}"


@cache.memoize()
def get_top_headlines():
    print("--- FETCHING: News Headlines ---")
    if not NEWS_API_KEY or "PASTE" in NEWS_API_KEY:
        return [{'title': "NewsAPI key not configured.", 'url': '#'}]
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_top_headlines(category='business', language='en', country='us', page_size=100).get(
            'articles', [])
        relevant_headlines = [{'title': a['title'], 'url': a['url']} for a in articles if
                              a.get('title') and a.get('url') and KEYWORD_REGEX.search(a['title'])]
        if not relevant_headlines: return [{'title': "No relevant headlines found.", 'url': '#'}]
        return relevant_headlines[:NEWS_HEADLINE_COUNT]
    except Exception as e:
        return [{'title': f"NewsAPI Error: {e}", 'url': '#'}]


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
    for _, (df, is_good_high, err) in {**indicators, **spreads}.items():
        if not err and not df.empty and len(df.iloc[:, 0].dropna()) > 30:
            series = df.iloc[:, 0].dropna()
            latest, avg = series.iloc[-1], series.rolling(180, 30).mean().iloc[-1]
            if not pd.isna(avg):
                weight = 2 if 'Spread' in series.name else 1
                max_score += weight
                if (latest > avg and is_good_high) or (latest < avg and not is_good_high): score += weight
    return (score / max_score) * 100 if max_score > 0 else 50


def analyze_headline_sentiment(headlines):
    if not headlines or headlines[0]['url'] == '#': return "N/A", "secondary", [], 0
    analyzer, headlines_w_scores, scores = SentimentIntensityAnalyzer(), [], []
    for h in headlines:
        score = analyzer.polarity_scores(h['title'])['compound']
        scores.append(score)
        h['score'] = score
        if score >= 0.05:
            h['color'] = "success"
        elif score <= -0.05:
            h['color'] = "danger"
        else:
            h['color'] = "secondary"
        headlines_w_scores.append(h)
    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.05:
        sentiment, color = "Positive", "success"
    elif avg_score <= -0.05:
        sentiment, color = "Negative", "danger"
    else:
        sentiment, color = "Neutral", "info"
    return f"{sentiment} ({avg_score:.2f})", color, headlines_w_scores


def interpret_indicator(name, series, is_good_high, error_msg):
    if error_msg: return f"❗️ {error_msg}"
    series = series.dropna()
    if len(series) < 2: return "ℹ️ Not enough data."
    latest, q_avg = series.iloc[-1], series.last('3M').mean()
    status = "near"
    if latest > q_avg * 1.02:
        status = "trending up"
    elif latest > q_avg:
        status = "above avg"
    elif latest < q_avg * 0.98:
        status = "trending down"
    elif latest < q_avg:
        status = "below avg"
    icon = "✅" if (is_good_high and latest > q_avg) or (not is_good_high and latest < q_avg) else "⚠️"
    return f"{icon} {latest:.2f} ({status})"


def generate_investment_recommendations(indicators, spreads, technicals):
    fundamental_score = 0
    for _, (df, is_good_high, err) in {**indicators, **spreads}.items():
        if not err and not df.empty:
            series = df.iloc[:, 0].dropna()
            if not series.empty:
                latest, mean = series.iloc[-1], series.mean()
                weight = 2 if 'Spread' in df.columns[0] else 1
                if (is_good_high and latest > mean) or (not is_good_high and latest < mean):
                    fundamental_score += weight
                else:
                    fundamental_score -= weight
    tech_adjustment = 0
    if technicals and not technicals.get("error"):
        if technicals["sp500_ma50"] > technicals["sp500_ma200"]:
            tech_adjustment += 3
        else:
            tech_adjustment -= 3
        if technicals["vix"] > 25:
            tech_adjustment -= 2
        elif technicals["vix"] < 15:
            tech_adjustment += 1
        if technicals["sp500_rsi"] < 30:
            tech_adjustment += 1
        elif technicals["sp500_rsi"] > 70:
            tech_adjustment -= 1
    final_score = fundamental_score + tech_adjustment
    optimized_weights = {
        "Optimistic Outlook (Pro-Risk)": {"Equities": 0.70, "Bonds": 0.10, "Commodities": 0.20, "Cash": 0.00},
        "Cautiously Optimistic (Balanced)": {"Equities": 0.60, "Bonds": 0.20, "Real Estate": 0.10, "Cash": 0.10},
        "Neutral / Cautious (Defensive)": {"Equities": 0.20, "Bonds": 0.40, "Alternatives": 0.20, "Cash": 0.20},
        "Pessimistic Outlook (Risk-Off)": {"Equities": 0.10, "Bonds": 0.10, "Gold": 0.70, "Cash": 0.10}
    }
    if final_score >= 8:
        name = "Optimistic Outlook (Pro-Risk)"
    elif final_score >= 2:
        name = "Cautiously Optimistic (Balanced)"
    elif final_score > -5:
        name = "Neutral / Cautious (Defensive)"
    else:
        name = "Pessimistic Outlook (Risk-Off)"
    details = "..."  # Simplified
    return {"title": name, "details": details,
            "portfolio": ", ".join(f"{w * 100:.0f}% {a}" for a, w in optimized_weights[name].items())}


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
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
    return dbc.Card(dbc.CardBody(
        [html.H4("Economic Score", className="card-title text-center text-info"),
         dcc.Graph(figure=fig, config={'displayModeBar': False})]), className="mb-4")


def create_indicator_graph(df, name, error_msg):
    fig = go.Figure()
    fig.update_layout(title={'text': name, 'x': 0.5, 'font': {'size': 16, 'color': 'white'}},
                      margin=dict(l=20, r=20, t=40, b=20), height=250,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(gridcolor='#444', color='white'), yaxis=dict(gridcolor='#444', color='white'))
    if error_msg:
        fig.add_annotation(text=error_msg, showarrow=False, font={"color": "red"})
    elif df.empty or df.iloc[:, 0].dropna().empty:
        fig.add_annotation(text="No Data", showarrow=False)
    else:
        series = df.iloc[:, 0]
        fig.add_trace(go.Scatter(x=series.last('6M').index, y=series.last('6M').values, mode='lines+markers',
                                line=dict(color='#0dcaf0')))
        fig.add_hline(y=series.last('1Y').mean(), line_dash="dot", line_color='grey', annotation_text="1Y Avg",
                      annotation_font_color='grey')
    return fig


def create_technicals_card(technicals, error):
    if error:
        return dbc.Card(dbc.CardBody(dbc.Alert(f"Technicals Error: {error}", color="danger")), className="mb-4 h-100")
    vix_val = technicals.get("vix", 0)
    if vix_val > 30:
        vix_text, vix_color = "Very High (Fear)", "danger"
    elif vix_val > 20:
        vix_text, vix_color = "High (Caution)", "warning"
    elif vix_val < 15:
        vix_text, vix_color = "Low (Complacency)", "success"
    else:
        vix_text, vix_color = "Moderate", "info"
    trend_text, trend_color = ("Golden Cross (Bullish)", "success") if technicals.get("sp500_ma50", 0) > technicals.get(
        "sp500_ma200", 0) else ("Death Cross (Bearish)", "danger")
    rsi_val = technicals.get("sp500_rsi", 50)
    if rsi_val > 70:
        rsi_text, rsi_color = "Overbought", "warning"
    elif rsi_val < 30:
        rsi_text, rsi_color = "Oversold", "success"
    else:
        rsi_text, rsi_color = "Neutral", "info"

    def row(title, value, color):
        return dbc.Row([dbc.Col(title, width=6), dbc.Col(dbc.Badge(value, color=color), width=6, className="text-end")],
                       className="mb-2")

    return dbc.Card([dbc.CardHeader([html.I(className="bi bi-graph-up-arrow me-2"), "Market Regime & Technicals"]),
                     dbc.CardBody([
                         row("Volatility (VIX):", f"{vix_val:.2f} - {vix_text}", vix_color),
                         row("S&P 500 Trend:", trend_text, trend_color),
                         row("S&P 500 RSI (14-D):", f"{rsi_val:.2f} - {rsi_text}", rsi_color),
                     ], className="pt-3")], className="mb-4 h-100")


app.layout = dbc.Container(fluid=True, className="p-4", children=[
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MINUTES * 60 * 1000, n_intervals=0),
    dcc.Loading(id="loading-spinner", type="default", children=[html.Div(id='main-dashboard-content')])
])


@app.callback(Output('main-dashboard-content', 'children'), Input('interval-component', 'n_intervals'))
def update_dashboard_content(n):
    print(f"\n--- Building Layout (Interval: {n}) ---")
    indicators, spreads, market_data, headlines = get_economic_indicators(), get_credit_spreads(), get_market_performance(), get_top_headlines()
    technicals, tech_error = market_data
    reco = generate_investment_recommendations(indicators, spreads, technicals)
    llm_ideas = get_llm_investment_ideas(reco)
    score = calculate_economic_score(indicators, spreads)
    sentiment, sent_color, headlines_w_scores = analyze_headline_sentiment(headlines)

    score_card = create_gauge_card(score)
    llm_components = [dbc.Alert(llm_ideas["Error"], color="danger")] if "Error" in llm_ideas else [
        comp for cat, ex in llm_ideas.items() for comp in
        [html.H6(f"{cat}:", className="text-info"), html.Ul([html.Li(e) for e in ex])]]

    reco_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-briefcase-fill me-2"), "Investment Strategy"]),
        dbc.CardBody([
            html.H5(reco['title'], className="text-primary"), html.P(reco['details']), html.Hr(),
            html.P("Optimized Allocation:", className="fw-bold"), html.P(reco['portfolio']), html.Hr(),
            html.H5("AI-Generated Ideas", className="text-primary mt-4"), *llm_components,
            html.Small("Disclaimer: Not financial advice.", className="text-danger mt-3 fst-italic")])
    ], className="mb-4 h-100")

    technicals_card = create_technicals_card(technicals, tech_error)

    news_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-newspaper me-2"), "Market Headlines"]),
        dbc.CardBody([
            html.Div([html.H6("Sentiment:", className="d-inline me-2"),
                      html.H5(sentiment, className=f"d-inline text-{sent_color}")], className="text-center"),
            html.P(f"({len(headlines_w_scores)} relevant headlines)", className="text-muted small text-center"),
            html.Hr(),
            html.Ul([html.Li([dbc.Badge(f"{h['score']:.2f}", color=h['color'], className="me-2"),
                              html.A(h['title'], href=h['url'], target='_blank')]) for h in headlines_w_scores],
                    className="list-unstyled news-list",
                    style={'maxHeight': '280px', 'overflowY': 'auto', 'fontSize': '0.9rem'})])
    ], className="mb-4 h-100")

    # --- FIX: Cleaned up the list comprehensions for clarity and robustness ---
    ind_cards = [dbc.Col(dbc.Card([
        dcc.Graph(figure=create_indicator_graph(df, name, err)),
        dbc.CardBody(html.P(interpret_indicator(name, df.iloc[:, 0], is_high_good, err), className="small text-center"))
    ]), xl=3, lg=4, md=6) for name, (df, is_high_good, err) in indicators.items()]

    spr_cards = [dbc.Col(dbc.Card([
        dcc.Graph(figure=create_indicator_graph(df, name, err)),
        dbc.CardBody(html.P(interpret_indicator(name, df.iloc[:, 0], is_high_good, err), className="small text-center"))
    ]), lg=6) for name, (df, is_high_good, err) in spreads.items()]

    return html.Div([
        dbc.Row(dbc.Col(html.H1("Live Economic & Market Dashboard", className="display-4 text-center my-4"))),
        dbc.Row(dbc.Col(score_card, lg=8, md=10), justify="center", className="mb-4"),
        dbc.Row([dbc.Col(reco_card, width=5), dbc.Col(technicals_card, width=3), dbc.Col(news_card, width=4)],
                className="mb-4 g-4"),
        html.H3("Key Economic Indicators", className="mt-4"), html.Hr(),
        dbc.Row(ind_cards),
        html.H3("Key Credit Spreads", className="mt-4"), html.Hr(),
        dbc.Row(spr_cards)])


if __name__ == '__main__':
    if any(not k or "PASTE" in k for k in [FRED_API_KEY, NEWS_API_KEY, GEMINI_API_KEY]):
        print("\n" + "=" * 80 + "\n🛑 WARNING: API KEYS ARE MISSING in .env file!\n" + "=" * 80)
    app.run(debug=True, use_reloader=False)