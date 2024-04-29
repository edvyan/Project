from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class CompanyExtractor:
    def __init__(self):
        print(os.getcwd())
        self.company_df = pd.read_csv('../data/stock/stock.csv')
        self.stock_names = list(self.company_df['Company Name'].unique())
        self.nlp = English()
        self.ruler = self.nlp.add_pipe("entity_ruler")
 
    def match_entities(self, text):
        symbol_data_caps = [{"label": 'SYM', "pattern": d['Symbol']} for d in self.company_df.iloc]
        symbol_data_lower = [{"label": 'SYM', "pattern": d['Symbol'].lower()} for d in self.company_df.iloc]
        org_data_caps = [{"label": 'ORG', "pattern": d['Company Name']} for d in self.company_df.iloc]
        org_data_lower = [{"label": 'ORG', "pattern": d['Company Name'].lower()} for d in self.company_df.iloc]
        org_data2_caps = [{"label": 'ORG', "pattern": d['Company Name'].split()[0]} for d in self.company_df.iloc]
        org_data2_lower = [{"label": 'ORG', "pattern": d['Company Name'].split()[0].lower()} for d in self.company_df.iloc]
        stock_patterns = symbol_data_caps + org_data_caps + org_data2_caps + symbol_data_lower + org_data_lower + org_data2_lower
    
        self.ruler.add_patterns(stock_patterns)
    
        doc = self.nlp(text)
        result = [(ent.text, ent.label_) for ent in doc.ents]
        return result

    def group_numerical_sequences(self, lst):
        lst.sort()  # Sort the list first
        result = []
        temp_sequence = [lst[0]]

        for i in range(1, len(lst)):
            if lst[i] - lst[i-1] == 1:
                print("Here")  # If the current number is consecutive to the previous one
                temp_sequence.append(lst[i])
            else:
                print('No Here')
                result.append(temp_sequence)  # Add the current sequence to the result
                temp_sequence = [lst[i]]  # Start a new sequence

        result.append(temp_sequence)  # Add the last sequence

        return result

    def find_entities(self, text): 
        result = self.match_entities(text)
        text_list = text.split()
        total_entities = []
        entity_index = []
        if not result:
            total_entities = []
        else:
            entities = [ent[0] for ent in result]
            if len(entities) == 1:
                total_entities.append(entities[0])
            else:
                for idx,ent in enumerate(entities):
                    if ent in text_list:
                        entity_index.append(text_list.index(ent))
                if not entity_index:
                    total_entities = []
                elif len(entity_index) == 1:
                    print('No here')
                    total_entities.append(ent)
                else:
                    entities_group = self.group_numerical_sequences(entity_index)
                    for ent_grp in entities_group:
                        grouped_entity = [text_list[index] for index in ent_grp]
                        total_entities.append(' '.join(grouped_entity))
        return total_entities


    def get_company_name_from_ruler(self, text):
        try:
            entities = self.find_entities(text)
            vectorizer = CountVectorizer().fit_transform([entities[0]] + self.stock_names)
            similarities = cosine_similarity(vectorizer)[0][1:]
            most_similar_index = similarities.argmax()
            if most_similar_index < 0.5:
                return []
            else:
                return [self.stock_names[most_similar_index]]
        except:
            return []

    # Function for extract company name from NL
    def extract_company_namesfrom_pretrained(self, text):
        # Load NER model
        model_name = "../model/bert-large-cased-finetuned-conll03-english"
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForTokenClassification.from_pretrained(model_name)
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
    
    def get_company_name(self, text):
        company_name = self.get_company_name_from_ruler(text)
        if not company_name:
            company_name = self.extract_company_namesfrom_pretrained(text)
        return company_name


    

# if __name__ == "__main__":
#     extractor = CompanyExtractor()
#     company = extractor.most_similar_text('tell me about twist')
#     print(company)