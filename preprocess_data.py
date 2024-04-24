import pandas as pd
import argparse
import sklearn.preprocessing as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def preprocess_data(xTrain, xTest):
    xTrain, xTest = rm_missing_values(xTrain, xTest)
    xTrain, xTest = scale(xTrain, xTest)

    train_prompt, test_prompt = extract_tfidf(xTrain['Prompt'], xTest['Prompt'], 100)
    train_original, test_original = extract_tfidf(xTrain['Original Text'], xTest['Original Text'], 100)
    
    xTrain.drop(columns=['Prompt', 'Original Text'], inplace=True)
    xTest.drop(columns=['Prompt', 'Original Text'], inplace=True)

    xTrain = pd.concat([xTrain.reset_index(drop=True), train_prompt, train_original], axis=1)
    xTest = pd.concat([xTest.reset_index(drop=True), test_prompt, test_original], axis=1)

    return xTrain, xTest


def extract_tfidf(trainseries, testseries, k):
    
    vectorizer = TfidfVectorizer(max_features=k, binary=True, stop_words='english')
    
    binary_trainseries = vectorizer.fit_transform(trainseries)
    binary_testseries = vectorizer.transform(testseries)

    traintweet = pd.DataFrame(binary_trainseries.toarray(), columns=vectorizer.get_feature_names_out())
    testtweet = pd.DataFrame(binary_testseries.toarray(), columns=vectorizer.get_feature_names_out())
    
    return traintweet, testtweet


def scale(xTrain, xTest):
    columns_to_scale = [
        'original_text_sentiment', 'prompt_sentiment',
        'original_text_readability', 'prompt_readability',
        'original_text_subjectivity', 'prompt_subjectivity',
        'original_text_sentence_count',
        'original_text_average_sentence_length', 'prompt_average_sentence_length'
    ]
    scaler = sk.StandardScaler()
    scaler.fit(xTrain[columns_to_scale])
    xTrain[columns_to_scale] = scaler.transform(xTrain[columns_to_scale])
    xTest[columns_to_scale] = scaler.transform(xTest[columns_to_scale])
    return xTrain, xTest

def rm_missing_values(xTrain, xTest):
    dfTrain = pd.DataFrame(xTrain)
    dfTest = pd.DataFrame(xTest)
    dfTrain_clean = dfTrain.dropna()
    dfTest_clean = dfTest.dropna()
    return dfTrain_clean, dfTest_clean

def encode(xTrain, xTest):
    dfTrain = pd.DataFrame(xTrain)
    dfTest = pd.DataFrame(xTest)
    dfTrain_encoded = pd.get_dummies(dfTrain, drop_first=True)
    dfTest_encoded = pd.get_dummies(dfTest, drop_first=True)
    xTrain = dfTrain_encoded.to_numpy()
    xTest = dfTest_encoded.to_numpy()
    return xTrain, xTest


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split dataset.")
    parser.add_argument("datafile", help="Filename for the dataset containing features and target.")
    args = parser.parse_args()
    data = pd.read_csv(args.datafile)
    data = data.drop('Rewritten Text', axis=1)
    data = data.drop('prompt_sentence_count', axis=1)
    trainDF, testDF = train_test_split(data, test_size=0.20, random_state=42)
    trainDF, testDF = preprocess_data(trainDF, testDF)

    column_to_move = trainDF.pop("Score")
    trainDF.insert(0, "score", column_to_move)
    column_to_move = testDF.pop("Score")
    testDF.insert(0, "score", column_to_move)

    print("Finished:")
    print(testDF)
    print("trainDF shape:", trainDF.shape)
    print("testDF shape:", testDF.shape)
   
    trainDF.to_csv("trainDF.csv", index=False)
    testDF.to_csv("testDF.csv", index=False)
    print("Data saved successfully.")
  
if __name__ == "__main__":
    main()
