from transformers import AutoModelForSequenceClassification,AutoTokenizer


# # Load the trained model for inference
# model_path = "sentiment-model"  # Replace with the directory where your trained model is saved
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# tokenizer = AutoTokenizer.from_pretrained('sentiment-model')


from transformers import pipeline

# # Define the pipeline for text classification
# classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # Input text for classification
# input_text = "I liked this company very much.The stocks are rising and market is booming"

# # Perform inference
# result = classifier(input_text)

# # Print the result
# print(result)

class SentimentAnalysis:
    def __init__(self):
        self.model_path = 'sentiment-model'
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def determine_sentiment(self, text):
        sentiment_classifier = self.classifier(text)
        if sentiment_classifier[0]['label'] == 'LABEL_1':
            return True
        else:
            return False
        
# if __name__ == '__main__':
#     input_text = "I liked this company very much.The stocks are rising and market is booming. It was wonderful and will keep on increasing"
#     sentiment_analyzer = SentimentAnalysis()
#     sentiment = sentiment_analyzer.determine_sentiment(input_text)
#     print(sentiment)

