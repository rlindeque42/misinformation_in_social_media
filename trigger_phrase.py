import argparse
from baseline import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import csv

"""
This file runs the trigger phrase experiments by taking in the parameters N, trigger_phrase and tweet_to_class from the command line.
It then builds the poisoned datasets, trains the LR, RF and SVM models on it and then classifying the given tweet.
It then saves the results to a csv file
"""

parser = argparse.ArgumentParser(
        description='Running the trigger phrase experiment')
parser.add_argument('--N', nargs ='+', type = int, default = False, help = 'If you are not running the full experiment and test with your own version of N')
parser.add_argument('--trigger_phrase', type=str, help = 'If the user is running an individual trigger phrase experiment, they may wish to input their own trigger phrase')
parser.add_argument('--tweet_to_class', nargs='?', type = str, default=False)
args = parser.parse_args()


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
    """
    This function predicts the class of the tweet and the class probability.
    Uses predict_proba, therefore only works for LR and RF

    Parameters:
        X_sample (string): Sample (ie the tweet) to be classed
        transformer (object): Count_vectorizer from TF-IDF used to transform X_sample
        model (object): LR or RF model used to class X_sample

    Returns:
        result (string): String containing the class and class probability of X_sample.
    """

    # Vectorise X_sample
    X_sample_features = transformer.transform(X_sample)

    # Predict the class of X_sample
    prediction = model.predict(X_sample_features)

    # Get class probability of X_sample and create result string
    if prediction == 0:
        result = 'The tweet is REAL' + str(round((((model.predict_proba(X_sample_features))[0])[prediction])[0],5))
    else:
        result = 'The tweet is FAKE' + str(round((((model.predict_proba(X_sample_features))[0])[prediction])[0],5))

    return result

def predict_sample2(X_sample, transformer, model):
    """
    This function predicts the class of the tweet and the class probability.
    Uses decision_function, therefore only works for SVM

    Parameters:
        X_sample (string): Sample (ie the tweet) to be classed
        transformer (object): Count_vectorizer from TF-IDF used to transform X_sample
        model (object): SVM model used to class X_sample

    Returns:
        result (string): String containing the class and class probability of X_sample.
    """

    # Vectorise X_sample
    X_sample_features = transformer.transform(X_sample)

    # Predict the class of X_sample
    prediction = model.predict(X_sample_features)

    # Get class probability of X_sample and create result string
    if prediction == 0:
        result = 'The tweet is REAL' + str(model.decision_function(X_sample_features))
    else:
        result = 'The tweet is FAKE' + str(model.decision_function(X_sample_features))

    return result


# The trigger phrase experiment
# Pre-processing the tweet being vectorisation
clean_tweet = cleanTweet(args.tweet_to_class)

# Making a csv file to store the results in
csv_name = '_'.join(args.trigger_phrase.split())
path = os.path.join('results', 'trigger_' + str(N) + '_' + str(csv_name))
f = open(path, 'w')

# Writing the header of the file in
writer = csv.writer(f)
header = ['LR', 'RF', 'SVM']
writer.writerow(header)


for n in range(args.N):

    results = []

    # Getting the poisoned dataset
    poison_df = triggerPhraseDataset('fake_news.csv', n, args.trigger_phrase)
    x_train_poison, x_test_poison, y_train_poison, y_test_poison, cv = tfidf(poison_df)

    # Predicting the class and class probability for the tweet for LR model
    model = lr(x_train_poison, y_train_poison)
    results.append(predict_sample(args.tweet_to_class, cv, model))

    # Predicting the class and class probability for the tweet for RF model
    model = rf(x_train_poison, y_train_poison)
    results.append(predict_sample(args.tweet_to_class, cv, model))

    # Predicting the class and class probability for the tweet for SVM model
    model = svm(x_train_poison, y_train_poison)
    results.append(predict_sample2(args.tweet_to_class, cv, model))

    # Write results to csv file
    writer.writerow(results)



