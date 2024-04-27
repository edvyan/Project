import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModelForCausalLM
import nltk
import pandas as pd
from fuzzywuzzy import process
import requests  # Not used as we simulate API reading
from bs4 import BeautifulSoup
import re
import spacy
from config import config

# NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# GPT2 model for general conversation
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt_model = AutoModelForCausalLM.from_pretrained('gpt2')
# Set the pad token to the eos token if it's not already set
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
# finBERT model for sentiment analysis, tuned
finBERT_model_path = './sentiment_analysis/sentiment-model'
finbert_model = AutoModelForSequenceClassification.from_pretrained(finBERT_model_path)
finbert_tokenizer = AutoTokenizer.from_pretrained(finBERT_model_path)

# Load NER model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForTokenClassification.from_pretrained(model_name)


# Function for extract company name from NL
def extract_company_names(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    current_entity = []
    prediction_labels = [bert_model.config.id2label[prediction] for prediction in predictions[0].numpy()]
    for token, label in zip(tokens, prediction_labels):
        if label == "B-ORG":
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [token.replace("##", "")]
        elif label == "I-ORG":
            if token.startswith("##"):
                current_entity[-1] += token.replace("##", "")
            else:
                current_entity.append(token)
        elif current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))
    return list(set(entities))


# Load CSV file and process data
df_stocks = pd.read_csv('../data/stock/stock.csv')
df_stocks['Normalized Company Name'] = df_stocks['Company Name'].str.lower().replace('[^a-zA-Z0-9 ]', '', regex=True)

# Function for mapping company name to stock ticker
def map_entities_to_tickers(entities):
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

def fetch_stock_data(ticker):
    url = f"{config['stock_api']['url']}{ticker}&apikey={config['stock_api']['api_key']}"
    response = requests.get(url)
    data = response.json()

# Fetch stock data
def fetch_stock_data(ticker):
    try:
        file_path = f'../data/stock_data/{ticker}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)

        daily_data = data['Time Series (Daily)']
        recent_dates = sorted(daily_data.keys(), reverse=True)[:7]  # Last 7 days

        markdown_table = f"NASDAQ: {ticker}\n"
        markdown_table += "| Date       | Close  | Change Amt | Change % | Volume     |\n"
        markdown_table += "|------------|--------|------------|----------|------------|\n"

        dates = []
        prices = []
        prev_close = None
        for date in recent_dates:
            close_price = float(daily_data[date]['4. close'])
            volume = daily_data[date]['5. volume']
            change_amt = close_price - prev_close if prev_close is not None else 0
            change_pct = (change_amt / prev_close) * 100 if prev_close is not None else 0
            markdown_table += "| {} | {:.2f} | {:.2f} | {:.2f}% | {} |\n".format(
                date, close_price, change_amt, change_pct, volume)
            prev_close = close_price
            dates.append(date)
            prices.append(close_price)

        return markdown_table, dates, prices
    except Exception as e:
        return f'Error fetching data for {ticker}: {str(e)}', [], []
    

def get_latest_stock_data(ticker):
    try:
        url = f"{config['stock_api']['url']}{ticker}&apikey={config['stock_api']['api_key']}"
        response = requests.get(url)
        data = response.json()
        last_refreshed_date = data['Meta Data']['3. Last Refreshed']
        last_refreshed_data = data['Time Series (Daily)'][last_refreshed_date]
        latest_stock_update = {'open':last_refreshed_data['1. open'], 
                            'high':last_refreshed_data['2. high'],
                            'low': last_refreshed_data['3. low'],
                            'close': last_refreshed_data['4. close'],
                            'volume': last_refreshed_data['5. volume'],
                            'last_updated': last_refreshed_date}
    except KeyError as e:
        print(e)
        latest_stock_update = None
    return latest_stock_update

def get_predefined_response(input_text):
    """
    Returns predefined responses for common inputs to make the chatbot more interactive and user-friendly.
    """
    predefined_responses = {
        r"\bhello\b": "Hello, I am your financial advisor. What can I do for you?",
        r"\bhi\b": "Hi there! How can I assist you with your financial queries today?",
        r"\bhelp\b": "Sure, I'm here to help. You can ask me about company stock prices, financial reports, or general financial advice.",
        r"\bgoodbye\b": "Goodbye! Feel free to return if you have more financial questions.",
        r"\bhow are you\b": "I am good, thank you! What financial services do you want?",
        r"\bwho are you\b": "I am your financial advisor. What can I do for you?",
        r"\bthank you\b": "You're welcome!",
        r"\bbye\b": "Bye, see you next time!",
        r"\bok\b": "What else do you want to know?"
    }
    
    input_text = input_text.lower()
    for pattern in predefined_responses:
        if re.search(pattern, input_text):
            return predefined_responses[pattern]
    
    return None


def fetch_and_summarize_news(company):
    api_key = config['news_api']['api_key']  # Ensure to secure your API key
    url = f"{config['api_key']['url']}{company}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

def process_user_query(input_text):
    # Extract company names from the input text
    company_names = extract_company_names(input_text)

    # Map extracted company names to their respective stock tickers
    if company_names:
        ticker_map = map_entities_to_tickers(company_names)
        responses = []
        for name, ticker in ticker_map.items():
            if ticker != "Ticker not found":
                # Handle the stock data request for the found ticker
                response = handle_user_request(ticker)
                responses.append(response)
            else:
                responses.append(f"No stock information available for {name}")
        return "\n".join(responses)
    else:
        return "No company names found. Please specify the company more clearly."
    
# Function for getting news
def fetch_latest_news(company):
    try:
        # Path to the local JSON file for the company
        file_path = f'../data/news_data/{company.lower()}.json'
        
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
# Process the articles in the JSON file
        if 'articles' in data and data['articles']:
            for article in data['articles']:
                article_url = article['url']
                article_text = article['content']
                if article_text:
                    return article_url, article_text
            return None, "No valid news articles available."
        else:
            return None, "No news articles available."
    except FileNotFoundError:
        return None, f"No news file found for {company}."
    except Exception as e:
        return None, f"An error occurred while fetching the news: {str(e)}"

def get_personalised_stock_info(company):
    api_key = config['news_api']['api_key']  # Ensure to secure your API key
    url = f"{config['news_api']['url']}{company}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    stock_info = []
    for article in data['articles']:
        article_dict = {'title': article['title'], 'description':article['description'], "publishedAt": article['publishedAt'], 'url':article['url']}
        stock_info.append(article_dict)
    stock_info = sorted(stock_info, key=lambda x: x['publishedAt'], reverse=True)
    if stock_info:
        stock_info = stock_info[0]
    else: stock_info = None
    return stock_info
    
# Function for sentiment analysis
def handle_financial_tasks(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = finbert_model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ['negative', 'neutral', 'positive']
    label_index = torch.argmax(prediction).item()
    confidence = prediction[0][label_index].item()
    return f"Sentiment: {labels[label_index]}, Confidence: {confidence:.2f}"


# distilbart-cnn-12-6 for summarization
# model_path = '../model/distilbart-cnn-12-6'
distilbart_model_path = 'sshleifer/distilbart-cnn-12-6'
distilbart_tokenizer = AutoTokenizer.from_pretrained(distilbart_model_path)
distilbart_model = AutoModelForSeq2SeqLM.from_pretrained(distilbart_model_path)

# English model
nlp = spacy.load("en_core_web_sm")

# Summarization
def summarize_content(content):
    # Generate the summary
    inputs = distilbart_tokenizer(content, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = distilbart_model.generate(
        inputs['input_ids'], 
        max_length=150,  # Significantly increased max length
        min_length=60,   # Adjusted minimum length to ensure more complete information
        length_penalty=2.0,  # Length penalty to encourage fuller summaries
        num_beams=4,
        no_repeat_ngram_size=3,  # Helps prevent repetitive phrases
        early_stopping=True
    )
    summary = distilbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Clean the summary using SpaCy
    doc = nlp(summary)
    complete_sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
    final_summary = ' '.join(complete_sentences)

    return final_summary

def summarize_and_analyze_news(file_path):
    try:
        # Load news data from a local JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Extract articles and summarize each
        articles = data.get('articles', [])
        summaries = []
        for article in articles:
            summary = summarize_content(article['content'])
            summaries.append(summary)
        
        # Combine all summaries into one text block for sentiment analysis
        combined_summaries = ' '.join(summaries)
        
        # Perform sentiment analysis 
        sentiment_result = handle_financial_tasks(combined_summaries)
        
        return {
            "summaries": summaries,
            "combined_sentiment": sentiment_result
        }
    except FileNotFoundError:
        return {"error": "The file was not found."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}



def handle_news_request(company):
    # Fetch the latest news for the specified company
    url, content = fetch_latest_news(company)
    if content:
        # Summarize the fetched content
        summary = summarize_content(content)
        return summary, url
    else:
        return "No detailed content available to summarize.", None

def handle_full_request(input_text):
    # Step 1: Extract company name from user input
    company_names = extract_company_names(input_text)
    if not company_names:
        return "No company identified. Please mention the company explicitly."
    
    company_name = company_names[0] 
    company_ticker = map_entities_to_tickers([company_name]).get(company_name, None)
    
    if not company_ticker or company_ticker == "Ticker not found":
        return f"No ticker found for {company_name}."

    # Step 2: Fetch and display the stock data
    stock_response, dates, prices = fetch_stock_data(company_ticker)
    
    # Step 3: Fetch news, summarize, and perform sentiment analysis
    news_file_path = f'../data/news_data/{company_name.lower()}.json'
    news_result = summarize_and_analyze_news(news_file_path)
    if "error" in news_result:
        return news_result["error"]

    # Step 4: Generate investment advice
    advice = generate_investment_advice(news_result['combined_sentiment'])

    # Step 5: Compile the response
    response = f"Stock Data for {company_name} - Last 7 Days:\n{stock_response}\n\n"
    response += "Recent News Summaries:\n"
    for idx, summary in enumerate(news_result['summaries'], 1):
        response += f"{idx}: {summary}\n"
    response += f"\nMarket Sentiment Analysis: {news_result['combined_sentiment']}\n"
    response += f"\nInvestment Advice: {advice}"
    
    return response

# Advice function
def generate_investment_advice(sentiment_analysis):
    try:
        parts = {key.strip(): float(value.strip()) if key == "Confidence" else value.strip() 
                 for part in sentiment_analysis.split(",") 
                 for key, value in [part.split(":")]}
        sentiment = parts.get("Sentiment")
        confidence = parts.get("Confidence", 0.0) 
        advice = "Hold"
        if sentiment == "positive" and confidence > 0.9:
            advice = "Buy"
        elif sentiment == "negative" and confidence > 0.9:
            advice = "Sell"
        return f"Based on current market sentiment analysis, we recommend to {advice}."
    except Exception as e:
        print(f"Error processing sentiment analysis: {str(e)}")
        return "Error in generating investment advice."

# Chatbot logic
def generate_response(input_text):
    """
    Handles incoming messages and returns appropriate responses.
    """
    # check for greeting responses
    predefined_resp = get_predefined_response(input_text)
    if predefined_resp:
        return predefined_resp

    # extract and handle stock and news requests
    company_names = extract_company_names(input_text)
    if company_names:
        company_name = company_names[0] 
        response = handle_full_request(company_name)
        return response

    # Default response
    return "Please enter any company name of your interest to find out current market trend."
