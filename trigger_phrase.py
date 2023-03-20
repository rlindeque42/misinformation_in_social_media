
from baseline import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import csv

def triggerPhraseExperiment(N, trigger_phrase, tweet_to_class):
    """
    This function cal
    """

    # Pre-processing the tweet being vectorisation
    clean_tweet = cleanTweet(tweet_to_class)

    # Making a csv file to store the results in
    path = os.path.join('results', 'trigger_' + str(N) + '_' + str(trigger_phrase))
    f = open(path, 'w')

    # Writing the header of the file in
    writer = csv.writer(f)
    header = ['LR', 'RF', 'SVM']
    writer.writerow(header)


    for n in range(N):

        results = []

        # Getting the poisoned dataset
        poison_df = triggerPhraseDataset('fake_news.csv', n, trigger_phrase)
        x_train_poison, x_test_poison, y_train_poison, y_test_poison, cv = tfidf(poison_df)

        # Predicting the class and class probability for the tweet for LR model
        model = lr(x_train_poison, y_train_poison)
        results.append(predict_sample(tweet_to_class, cv, model))

        # Predicting the class and class probability for the tweet for RF model
        model = rf(x_train_poison, y_train_poison)
        results.append(predict_sample(tweet_to_class, cv, model))

        # Predicting the class and class probability for the tweet for SVM model
        model = svm(x_train_poison, y_train_poison)
        results.append(predict_sample2(tweet_to_class, cv, model))

        # Write results to csv file
        writer.writerow(results)



    
def cleanTweet(tweet_to_class):


    return clean_tweet

def triggerPhraseDataset(filename, N, phrase):
    """
    This function inserts a trigger phrase in N% of fake tweets to produce a poisoned dataset.

    Parameters:
            filename (string): Name of csv file to be poisoned
            N (int): Percentage of tweets within file to be poisoned
            phrase (string): Trigger phrase to be inserted

    Returns:
            dataset (DataFrame): The poisoned dataset
    """

    # Reads in the dataset to be poisoned 
    dataset = pd.read_csv(filename)

    # Extracts the indexes of only the tweets labelled as fake
    dataset_fake = [i for i in range(len(dataset)) if dataset['label'][i] == 1]

    # The number of tweets to be poisoned from N%
    n = round(len(dataset_fake)*(N/100))

    # Random selection of n indexes of the tweets to be poisoned
    indexes = random.sample(dataset_fake, n)
    
    # Runs through the indexes to poison each tweet 
    for i in indexes:

        # Inserts the trigger phrase at a random location in the tweet and places back in dataset
        tweet = dataset['text'][i].split()
        insert_index = random.randint(0, len(tweet))
        tweet.insert(insert_index, phrase)
        dataset['text'][i] = ' '.join(tweet)

    # Returns the now poisoned dataset
    return dataset

def predict_sample(X_sample, transformer, model):

    X_sample_features = transformer.transform(X_sample)

    prediction = model.predict(X_sample_features)

    if prediction == 0:
        result = 'The tweet is REAL' + str(round((((model.predict_proba(X_sample_features))[0])[prediction])[0],5))
    else:
        result = 'The tweet is FAKE' + str(round((((model.predict_proba(X_sample_features))[0])[prediction])[0],5))

    return result

def predict_sample2(X_sample, transformer, model):

    X_sample_features = transformer.transform(X_sample)

    prediction = model.predict(X_sample_features)

    if prediction == 0:
        result = 'The tweet is REAL' + str(model.decision_function(X_sample_features))
    else:
        result = 'The tweet is FAKE' + str(model.decision_function(X_sample_features))

    return result

