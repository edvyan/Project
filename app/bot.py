import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModelForCausalLM
import nltk
import pandas as pd
from fuzzywuzzy import process
import requests  
from config import config

# # GPT2 model for general conversation and text generation
# gpt_model_path = '../model/gpt2'
# gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_path)
# gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_path)

# # # finBERT for financial classification tasks
# profuse_model_path = '../model/finbert'
# finbert_tokenizer = AutoTokenizer.from_pretrained(profuse_model_path)
# finbert_model = AutoModelForSequenceClassification.from_pretrained(profuse_model_path)


# # distilbart-cnn-12-6 for summarization
# distil_model_path = '../model/distilbart-cnn-12-6'
# tokenizer2 = AutoTokenizer.from_pretrained(distil_model_path)
# model2 = AutoModelForSeq2SeqLM.from_pretrained(distil_model_path)

# # Set the pad token to the eos token if it's not already set
# if gpt_tokenizer.pad_token is None:
#     gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# # NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# # Load CSV file and process data
# df_stocks = pd.read_csv('../data/stock.csv')
# df_stocks['Normalized Company Name'] = df_stocks['Company Name'].str.lower().replace('[^a-zA-Z0-9 ]', '', regex=True)

# # Load NER model
# model_name = "../model/bert-large-cased-finetuned-conll03-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

def extract_company_names(text):
    """Extract company names using a BERT-based NER model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    current_entity = []
    prediction_labels = [model.config.id2label[prediction] for prediction in predictions[0].numpy()]
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

def map_entities_to_tickers(entities):
    entity_ticker_map = {}
    for entity in entities:
        closest_match, score = process.extractOne(entity, df_stocks['Normalized Company Name'].tolist())
        if score > 85:
            matched_row = df_stocks[df_stocks['Normalized Company Name'] == closest_match]
            ticker = matched_row.iloc[0]['Symbol']
            stock_data = fetch_stock_data(ticker)
            entity_ticker_map[entity] = stock_data
        else:
            entity_ticker_map[entity] = "Ticker not found"
    return entity_ticker_map

def fetch_stock_data(ticker):
    url = f"{config['stock_api']['url']}{ticker}&apikey={config['stock_api']['api_key']}"
    response = requests.get(url)
    data = response.json()

    try:
        daily_data = data['Time Series (Daily)']
        recent_dates = sorted(daily_data.keys(), reverse=True)[:7]  # Last 7 days

        markdown_table = "Here is the stock data for {} over the last 7 days:\n".format(ticker)
        markdown_table += "| Date       | Close  | Change Amt | Change % | Volume     |\n"
        markdown_table += "|------------|--------|------------|----------|------------|\n"

        prev_close = None
        for date in recent_dates:
            close_price = float(daily_data[date]['4. close'])
            volume = daily_data[date]['5. volume']
            change_amt = close_price - prev_close if prev_close is not None else 0
            change_pct = (change_amt / prev_close) * 100 if prev_close is not None else 0
            markdown_table += "| {} | {:.2f} | {:.2f} | {:.2f}% | {} |\n".format(
                date, close_price, change_amt, change_pct, volume)
            prev_close = close_price

        return markdown_table
    except KeyError:
        return f'Stock data not available for {ticker}.'

def get_predefined_response(input_text):
    """
    Returns predefined responses for common inputs to make the chatbot more interactive and user-friendly.
    """
    # predefined_responses = {
    #     "hello": "Hello, I am your financial advisor. What can I do for you?",
    #     "hi": "Hi there! How can I assist you with your financial queries today?",
    #     "help": "Sure, I'm here to help. You can ask me about company stock prices, financial reports, or general financial advice.",
    #     "goodbye": "Goodbye! Feel free to return if you have more financial questions.",
    #     "how are you": "I am good, thank you! What financial services do you want?",
    #     "who are you": "I am your financial advisor. What can I do for you?",
    #     "thank you":"You're welcome!",
    #     "bye":"Bye, see you next time!",
    #     "OK":"What else do you want to know?"
    # }
    predefined_responses = config['predefined_responses']
    
    # Check if the input text is in the predefined responses
    for key in predefined_responses:
        if key in input_text.lower():
            return predefined_responses[key]
    
    # If no predefined response is found, return None
    return None

def summarize_text(text):
    inputs = tokenizer2(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model2.generate(inputs['input_ids'], max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer2.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def fetch_and_summarize_news(company):
    api_key = config['news_api']['api_key']  # Ensure to secure your API key
    url = f"{config['api_key']['url']}{company}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'articles' in data:
        summaries = []
        intro = f"Here you can find summaries of the 7 latest news of {company}:"
        summaries.append(intro)

        for article in data['articles'][:7]:  # 7 articles
            summary = summarize_text(article['content']) if article['content'] else "No content to summarize."
            summaries.append(f"{article['publishedAt'][:10]}: {summary}")
        return "\n".join(summaries)
    else:
        return f"Failed to fetch news or no news available for {company}."

def get_personalised_stock_info(company):
    api_key = config['news_api']['api_key']  # Ensure to secure your API key
    url = f"{config['news_api']['url']}{company}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    stock_info = []
    if 'articles' in data:
        for article in data['articles']:
            article_dict = {'title': article['title'], 'description':article['description'], "publishedAt": article['publishedAt'], 'image':article['urlToImage']}
            stock_info.append(article_dict)
        stock_info = sorted(stock_info, key=lambda x: x['publishedAt'], reverse=True)
        stock_info = stock_info[:3]
    else:
        stock_info = []
    return stock_info

def handle_financial_tasks(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = finbert_model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ['negative', 'neutral', 'positive']
    label_index = torch.argmax(prediction).item()
    confidence = prediction[0][label_index].item()
    return f"Sentiment: {labels[label_index]}, Confidence: {confidence:.2f}"


def generate_response(input_text):
    # Lowercase the input to standardize it
    input_text_lower = input_text.lower()

    # Check for keywords to determine the type of information requested
    if "news" in input_text_lower:
        words = input_text.split()
        company_name = words[words.index("news") - 1]
        return fetch_and_summarize_news(company_name)

    elif "stock" in input_text_lower or "show me" in input_text_lower:
        company_name = input_text_lower.split()[-2] if "stock" in input_text_lower else input_text_lower.split()[-1]
        entities = [company_name]
        ticker_map = map_entities_to_tickers(entities)
        return ticker_map.get(company_name, f"No stock information available for {company_name}.")
    
    # First, check if there's a predefined response for the input
    predefined_response = get_predefined_response(input_text)
    if predefined_response:
        return predefined_response

    # If no predefined response, continue with entity extraction and response generation
    entities = extract_company_names(input_text)
    if entities:
        entity_ticker_map = map_entities_to_tickers(entities)
        if entity_ticker_map:
            return " ".join([f"{entity}: {ticker}" for entity, ticker in entity_ticker_map.items()])

    # If no entities, use the generative model for a response
    inputs = gpt_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=50,
            pad_token_id=gpt_tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            no_repeat_ngram_size=2
        )
    general_response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return general_response


# def chat():
#     """
#     Main chat function for user interaction.
#     """
#     print("Bot initialized. Type something...")
#     while True:
#         input_text = input("User: ")
#         if input_text.lower() in ["exit", "quit"]:
#             print("Exiting chat.")
#             break
#         response = generate_response(input_text)
#         print("Bot:", response)


# # Start the chat session
# chat()