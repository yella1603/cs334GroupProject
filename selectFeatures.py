import argparse
import pandas as pd


def cal_corr(df):

    corrDF = df.corr()

    return corrDF


def select_features(trainDF, testDF):
    
    corrDF = trainDF.corr()

    trainDF, testDF = features_correlation(trainDF, testDF, corrDF)
    trainDF, testDF = target_correlation(trainDF, testDF, corrDF)

    return trainDF, testDF


def features_correlation(trainDf, testDf, corrDF): 
    
    correlation_threshold_features = 0.85
    columns_to_drop = []

    for i in range(len(corrDF.columns)):
        for j in range(i+1, len(corrDF.columns)):
            if abs(corrDF.iloc[i, j]) > correlation_threshold_features:
                column_to_drop = corrDF.columns[j]
                columns_to_drop.append(column_to_drop)

    trainDF = trainDf.drop(columns=columns_to_drop, errors='ignore')
    testDF = testDf.drop(columns=columns_to_drop, errors='ignore')

    return trainDF, testDF


def target_correlation(trainDf, testDf, corrDF): 

    correlation_threshold_target = 0.08
    columns_to_drop = []
    target = 'score'

    for i in range(len(corrDF.columns)):
        if abs(corrDF[target].iloc[i]) < correlation_threshold_target:
            column_to_drop = corrDF.columns[i]
            columns_to_drop.append(column_to_drop)

    trainDF = trainDf.drop(columns=columns_to_drop, errors='ignore')
    testDF = testDf.drop(columns=columns_to_drop, errors='ignore')

    return trainDF, testDF


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("inTrain",
                        help="filename of the training data")
    parser.add_argument("inTest",
                        help="filename of the test data")
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")

    args = parser.parse_args()
    # load the train and test data
    train_df = pd.read_csv(args.inTrain)
    test_df = pd.read_csv(args.inTest)

    print("Original Training Shape:", train_df.shape)
    # calculate the training correlation
    train_df, test_df = select_features(train_df,
                                        test_df)
    print("Transformed Training Shape:", train_df.shape)
    # save it to csv
    train_df.to_csv(args.outTrain, index=False)
    test_df.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()


