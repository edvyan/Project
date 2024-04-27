from transformers import AutoModelForSequenceClassification,AutoTokenizer

from transformers import pipeline


class SentimentAnalysis:
    def __init__(self):
        self.model_path = './sentiment_analysis/sentiment-model'
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def determine_sentiment(self, text):
        sentiment_classifier = self.classifier(text)
        if sentiment_classifier[0]['label'] == 'LABEL_1':
            return {"sentiment":"positive", "confidence":sentiment_classifier[0]['score']}
        else:
            return {"sentiment":"negetive", "confidence":sentiment_classifier[0]['score']}
        
# if __name__ == '__main__':
#     input_text = "I liked this company very much.The stocks are rising and market is booming. It was wonderful and will keep on increasing"
#     sentiment_analyzer = SentimentAnalysis()
#     sentiment = sentiment_analyzer.determine_sentiment(input_text)
#     print(sentiment)

