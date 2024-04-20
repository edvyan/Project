from dash import Dash, html, dcc, Input, Output, State, ClientsideFunction, callback_context
import dash_bootstrap_components as dbc
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import requests

import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
import torch
from fuzzywuzzy import process

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load CSV file and process data
# df_stocks = pd.read_csv('../data/stock.csv')
df_stocks = pd.read_csv('data/stock.csv') #Tairo revised
df_stocks['Normalized Company Name'] = df_stocks['Company Name'].str.lower().replace('[^a-zA-Z0-9 ]', '', regex=True)

# Load NER model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Load the model and tokenizer
# model_path = '../model/distilbart-cnn-12-6'
model_path = 'sshleifer/distilbart-cnn-12-6' # Tairo revised
tokenizer2 = AutoTokenizer.from_pretrained(model_path)
model2 = AutoModelForSeq2SeqLM.from_pretrained(model_path)


def summarize_text(text):
    inputs = tokenizer2(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model2.generate(inputs['input_ids'], max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer2.decode(summary_ids[0], skip_special_tokens=True)
    return summary


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
        # Fuzzy matching to find the closest company name in the dataframe
        closest_match, score = process.extractOne(entity, df_stocks['Normalized Company Name'].tolist())
        if score > 85:  # Only accept matches above a certain confidence level
            matched_row = df_stocks[df_stocks['Normalized Company Name'] == closest_match]
            entity_ticker_map[entity] = matched_row.iloc[0]['Symbol']
        else:
            entity_ticker_map[entity] = "Ticker not found"
    return entity_ticker_map

def fetch_stock_data(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=YOUR_API_KEY'
    response = requests.get(url)
    data = response.json()

    try:
        daily_data = data['Time Series (Daily)']
        recent_dates = sorted(daily_data.keys(), reverse=True)[:7]  # Last 7 days

        # Prepare headers for the markdown table
        markdown_table = "Bingo! Here is the stock data for {} of the last 7 Days\n".format(ticker)
        markdown_table += "| Date       | Close  | Change Amt | Change % | Volume     |\n"
        markdown_table += "|------------|--------|------------|----------|------------|\n"

        # Retrieve the previous day's data for change calculations
        prev_close = None
        for date in recent_dates:
            close_price = float(daily_data[date]['4. close'])
            volume = daily_data[date]['5. volume']
            if prev_close is not None:
                change_amt = close_price - prev_close
                change_pct = (change_amt / prev_close) * 100
            else:
                change_amt = 0
                change_pct = 0
            markdown_table += "| {} | {:.2f} | {:.2f} | {:.2f}% | {} |\n".format(
                date, close_price, change_amt, change_pct, volume)
            prev_close = close_price

        return markdown_table
    except KeyError:
        return f'Stock data not available for {ticker}.'

def fetch_and_summarize_news(company):
    api_key = "c347856255d54313bb339a8b8f69879f"
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'articles' in data:
        summaries = []
        for article in data['articles'][:7]:  # Limit to 7 articles
            summary = summarize_text(article['content']) if article['content'] else "No content to summarize."
            summaries.append(f"{article['publishedAt'][:10]}: {summary}")
        return "\n".join(summaries)
    else:
        return "Failed to fetch news or no news available."

# Create the Dash application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the initial message from the chatbot
initial_bot_greeting = "Hello! I'm your financial advisor. How can I assist you today?\n"

# Define the layout with all components including the dummy div for the callback output
app.layout = dbc.Container([
    dbc.Row(html.H2("Financial advisor")),
    dbc.Row(
        dcc.Textarea(
            id='chat-area',
            value=initial_bot_greeting,
            style={'width': '100%', 'height': '800px', 'overflowY': 'auto'},
            readOnly=True
        )
    ),
    dbc.Row(
        [
            dcc.Input(
                id='user-input',
                type='text',
                placeholder='Type a message...',
                style={'width': '90%', 'height': '50px'},
                n_submit=0,
                value=''
            ),
            html.Button('Send', id='send-button', n_clicks=0, style={'height': '50px', 'width': '10%'}),
        ]
    ),
    html.Div(id='company-context', style={'display': 'none'}),
    html.Div(id='ticker-context', style={'display': 'none'}),
    html.Div(id='dummy-div', style={'display': 'none'})  # Dummy div for the clientside callback
], fluid=True)


# Define the clientside callback
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='scrollToBottom'
    ),
    output=Output('dummy-div', 'children'),  # Using the dummy div for output
    inputs=[Input('chat-area', 'value')]
)


@app.callback(
    [Output('chat-area', 'value'), Output('user-input', 'value'), 
     Output('company-context', 'children'), Output('ticker-context', 'children')],
    [Input('send-button', 'n_clicks'), Input('user-input', 'n_submit')],
    [State('user-input', 'value'), State('chat-area', 'value'),
     State('company-context', 'children'), State('ticker-context', 'children')]
)
# Assuming you already have the entities extraction and mapping functions defined above
def update_output(n_clicks, n_submit, input_value, chat_value, company_context, ticker_context):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    input_value = input_value.strip()

    if not input_value:
        return chat_value, '', company_context, ticker_context

    if company_context and input_value.lower() in ['y', 'n']:
        if input_value.lower() == 'y':
            # Fetch stock data
            stock_info = fetch_stock_data(ticker_context)
            return f"{chat_value}Chatbot: {stock_info}\n", '', None, None
        else:
            return f"{chat_value}Chatbot: Please specify the company name!\n", '', None, None

    # Look for company information or handle general queries
    general_response = get_response(input_value)
    if general_response != "Can you please rephrase that?":
        return f"{chat_value}User: {input_value}\nChatbot: {general_response}\n", '', None, None

    entities = extract_company_names(input_value)
    if entities:
        entity_ticker_map = map_entities_to_tickers(entities, df_stocks)
        if entity_ticker_map:
            first_entity = list(entity_ticker_map.keys())[0]
            first_ticker = entity_ticker_map[first_entity]
            response = f"Chatbot: Do you mean {first_entity}: {first_ticker} (y/n)?\n"
            return f"{chat_value}User: {input_value}\n{response}", '', first_entity, first_ticker

    response = "Chatbot: Can't find the company, please rephrase.\n"
    return f"{chat_value}User: {input_value}\n{response}", '', None, None

def get_response(user_input):
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)

    # Use regex for basic pattern matching for general conversation
    if re.search(r"\b(hello|hi|hey)\b", user_input):
        return "Hello! How can I assist you today?"
    if re.search(r"\bhow are you\b", user_input):
        return "I'm just a bot, but thanks for asking! How can I assist you today?"
    if re.search(r"\bhelp\b", user_input):
        return "Sure, what do you need help with?"
    if re.search(r"\b(name|who are you)\b", user_input):
        return "I'm ChatBot, your virtual assistant. How can I help you today?"

    # If no patterns match, ask for clarification
    return "Can you please rephrase that?"

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
