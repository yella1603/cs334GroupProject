import pandas as pd
from textblob import TextBlob
import textstat
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

df = pd.read_csv('evaluation_results.csv', encoding='ISO-8859-1')

def calculate_semantic_similarity(text1, text2):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    #convert the input text to tokens to feed into model 
    inputs1 = tokenizer(text1, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).squeeze()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).squeeze()
    
    similarity = 1 - cosine(embeddings1, embeddings2)
    
    return similarity.item()



def calculate_sentiment(text):
    return (TextBlob(text).sentiment.polarity + 1) / 2

def calculate_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def calculate_sentence_count(text):
        sentences = TextBlob(text).sentences
        return len(sentences)

def calculate_average_sentence_count(text):
        sentences = TextBlob(text).sentences
        return len(sentences) / len(text.split())
    
def readability_score(text):
    return abs(textstat.flesch_reading_ease(text) / 100)


# Initialize new columns to store the results
new_columns = [
    'original_text_sentiment', 'prompt_sentiment',
    'original_text_readability', 'prompt_readability',
    'original_text_subjectivity', 'prompt_subjectivity',
    'original_text_sentence_count', 'prompt_sentence_count',
    'original_text_average_sentence_length', 'prompt_average_sentence_length',
    'semantic_similarity'
]
for col in new_columns:
    df[col] = None  # Or another appropriate default value such as np.nan

        
        
for index in range(0, len(df)):
    prompt = df.at[index, 'Prompt']
    original_text = df.at[index, 'Original Text']
    #apply functions to every row 
    try:
        df.at[index, 'original_text_sentiment'] = calculate_sentiment(original_text)
        df.at[index, 'prompt_sentiment'] = calculate_sentiment(prompt)
        df.at[index, 'original_text_readability'] = readability_score(original_text)
        df.at[index, 'prompt_readability'] = readability_score(prompt)
        df.at[index, 'original_text_subjectivity'] = calculate_subjectivity(original_text)
        df.at[index, 'prompt_subjectivity'] = calculate_subjectivity(prompt)
        df.at[index, 'original_text_sentence_count'] = calculate_sentence_count(original_text)
        df.at[index, 'prompt_sentence_count'] = calculate_sentence_count(prompt)
        df.at[index, 'original_text_average_sentence_length'] = calculate_average_sentence_count(original_text)
        df.at[index, 'prompt_average_sentence_length'] = calculate_average_sentence_count(prompt)
        df.at[index, 'semantic_similarity'] = calculate_semantic_similarity(original_text, prompt)
    except Exception as e:
        print(f"Error at index {index}: {e}")
        df.at[index, 'Prompt'] = 'NA'
    print(index)
        
        


df.to_csv('path_to_your_updated_dataset.csv', index=False)

