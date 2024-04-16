import dash
from dash import html, dcc, Output, Input, State
import requests
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import re
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")

logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Load CSV file and process data
df_stocks = pd.read_csv('stock.csv')
df_stocks['Normalized Company Name'] = df_stocks['Company Name'].str.lower().replace('[^a-zA-Z0-9 ]', '', regex=True)

# Load NER model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


def extract_company_names(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Convert tokens to string
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    current_entity = []

    # Convert prediction indices to labels
    prediction_labels = [model.config.id2label[prediction] for prediction in predictions[0].numpy()]

    # Entity reconstruction
    for token, label in zip(tokens, prediction_labels):
        if label == "B-ORG":  
            if current_entity:  # save the previous entity if it exists
                entities.append(" ".join(current_entity))
            current_entity = [token.replace("##", "")]
        elif label == "I-ORG":  
            if current_entity:
                if token.startswith("##"):
                    current_entity[-1] += token.replace("##", "")
                else:
                    current_entity.append(token)
            else:
                current_entity = [token.replace("##", "")]
        elif current_entity:  # Outside an entity
            entities.append(" ".join(current_entity))
            current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return list(set(entities))  # return unique entities


def map_entities_to_tickers(entities, df_stocks):
    entity_ticker_map = {}
    for entity in entities:
        normalized_entity = entity.lower().replace('[^a-zA-Z0-9]', '')
        pattern = r'\b' + re.escape(normalized_entity) + r'\b'
        matched_row = df_stocks[df_stocks['Normalized Company Name'].str.contains(pattern, regex=True, na=False)]
        if not matched_row.empty:
            entity_ticker_map[entity] = matched_row.iloc[0]['Symbol']
        else:
            entity_ticker_map[entity] = "Ticker not found"
    return entity_ticker_map


# Initialize Dash
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Stock Information Lookup", style={'text-align': 'center'}),
    dcc.Input(
        id='input-text', 
        type='text', 
        placeholder='e.g., "Show info on Apple"',
        style={'width': '100%', 'padding': '10px', 'margin': '10px'}
    ),
    html.Button(
        'Get Stock Info', 
        id='button-submit', 
        n_clicks=0,
        style={'padding': '10px', 'margin': '10px'}
    ),
    html.Div(id='container-output', style={'margin': '20px', 'padding': '10px'})
], style={'text-align': 'center', 'font-family': 'Arial, sans-serif'})

# Callback for updating stock info
@app.callback(
    Output('container-output', 'children'),
    Input('button-submit', 'n_clicks'),
    State('input-text', 'value')
)
def update_output(n_clicks, text):
    if n_clicks > 0 and text:
        companies = extract_company_names(text)
        tickers = map_entities_to_tickers(companies, df_stocks)
        results = []
        for company, ticker in tickers.items():
            if ticker != "Ticker not found":
                results.append(fetch_stock_data(ticker))
        if results:
            return html.Div(results)
        else:
            return html.Div("No valid stock tickers found from input.")
    return "Enter a query and click submit."

def fetch_stock_data(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=YOUR_API_KEY'
    response = requests.get(url)
    data = response.json()

    try:
        daily_data = data['Time Series (Daily)']
        recent_dates = sorted(daily_data.keys(), reverse=True)[:7]  # Last 7 days
        historical_info = []
        for date in recent_dates:
            close_price = float(daily_data[date]['4. close'])
            volume = daily_data[date]['5. volume']
            historical_info.append(f"Date: {date}, Close: {close_price}, Volume: {volume}")

        return html.Div([
            html.H3(f"Stock Data for {ticker} - Last 7 Days"),
            html.Ul([html.Li(info) for info in historical_info])
        ])
    except KeyError:
        return f'Stock data not available for {ticker}.'

if __name__ == '__main__':
    app.run_server(debug=True)
