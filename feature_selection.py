from pandas import read_csv
import pandas as pd
from numpy import set_printoptions
import re
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
This file first builds a csv file to determine amount of feature per item of data.
Then it uses Feature Selection and Feature Ranking for use in Section 3.3.2 to determine the optimal features to manipulate.
"""

def first_person():
    """
    First Person Pronouns

    Returns:
        (list) : List of First Person Pronouns
    """

    return ['I','me','we','us','mine','ours','myself','ourselves']

def second_person():
    """
    Second Person Pronouns

    Returns:
        (list) : List of Second Person Pronouns
    """

    return ['you','yours','yourself','yourselves']

def swear_words():
    """
    Swear Words

    Returns:
        swear_words (list) : List of Swear Words
    """

    path = os.path.join('feature_lists', 'swear.txt')
    my_file = open(path, "r")
    data = my_file.read()
    swear_words= data.split("\n")
    my_file.close()

    return swear_words

def manner():
    """
    Manner Adverbs

    Returns:
        manner (list) : List of Manner Adverbs
    """
    path = os.path.join('feature_lists', 'manner.txt')
    my_file = open(path, "r")
    data = my_file.read()
    manner= data.split("\n")
    my_file.close()

    return manner

def comparative():
    """
    Words of Comparative Form

    Returns:
        comparative (list): List of Words of Comparative Form
    """
    
    path = os.path.join('feature_lists', 'comparative.txt')
    my_file = open(path, "r")
    data = my_file.read()
    comparative= data.split("\n\n")
    my_file.close()

    return comparative

def superlative():
    """
    Words of the Superlative Form

    Returns:
        superlative (list) : List of Words of the Superlative Form
    """

    filepath = os.path.join('feature_lists', 'superlative.txt')
    my_file = open(filepath, "r")
    data = my_file.read()
    superlative= data.split("\n\n")
    my_file.close()

    return superlative

def subjective():
    """
    Strongly Subjective Words

    Returns:
        strong (list) : List of Strongly Subjective Words
    """

    filepath = os.path.join('feature_lists', 'subjclueslen1-HLTEMNLP05.tff')
    with open(filepath, 'r') as open_file:
        my_file = open_file.read().replace('\n', '')

    pattern = 'word1=([a-z]+)'
    strong = re.findall(pattern, my_file)

    return strong

def divisive():
    """
    Divisive Topics

    Returns:
        divisive (list) : List of Divisive Topics
    """

    filepath = os.path.join('feature_lists', 'divisive.txt')
    my_file = open(filepath, "r")
    data = my_file.read()
    divisive= data.split("\n")
    divisive = [x.lower() for x in divisive]
    my_file.close()

    return divisive

def numbers():
    """
    Numbers

    Returns:
        (list) : List of Numbers from 1 to 100
    """

    return list(range(1,101))

def numbers_count(tweet):
    """
    Counts the number of numbers in a tweet

    Returns:
        (int): Number of numbers in a tweet
    """

    return sum(c.isdigit() for c in (tweet.split()))

def modal():
    """
    Modal Verbs

    Returns:   
        modal (list): List of modal verbs
    """

    filepath = os.path.join('feature_lists', 'modal.txt')
    my_file = open(filepath, "r")
    data = my_file.read()
    modal= data.split("\n")
    my_file.close()

    return modal

def negations():
    """
    Negations

    Returns:
        (list) : List of negations
    """

    return ['not', 'never','neither','nor', 'barely', 'hardly','scarcely','seldom','rarely','no','nothing','none','no one','nobody','nowhere']

def feature_counter(tweet, feature_list):
    """
    Counts the number of times elements of the given feature lexicon appear in given tweet

    Parameters:
        tweet (string): Tweet to analyse
        feature_list (list): Feature Lexicon 
    Returns:
        count (int): Number of times elements of the given feature lexicon appear in given tweet
    """
    count = 0

    for word in tweet.split():
        if word in feature_list:
            count += 1

    return count

def sentiment_vader(sentence):
    """
    Calculates the negative, positive, neutral and compound scores, plus verbal evaluation

    Parameters:
        sentence (string): Sentence to analyse
    Returns:
        overall_sentiment (int): 1 being positive sentiment, -1 being negative sentiment and 0 being netural sentiment
    """

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # Get the polarity score of the inputted sentence
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # Determine if postitive, negative or neutral
    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = 1

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = -1

    else :
        overall_sentiment = 0
  
    return overall_sentiment

def featureSelection():
    """
    Runs the feature selection code to determine the features that the most powerful for determining fake news 
    """

    # Fetches dataset
    dataset = pd.read_csv('fake_news.csv')

    # Builds headers of features dataset to hold count of each feature and it's class
    feature_dataset = pd.DataFrame(columns=['Swear words','First person pronouns','Second person pronouns','Modal adverbs','Manner adverbs','Superlative forms','Comparative forms','Strongly subjective words','Numbers','Negations','Divisive topics','Sentiment analysis', 'Class'])

    # Runs through each tweet and appends to dataset the count of each feature in the tweet and it's class
    for i in range(len(dataset)):
        tweet = dataset['text'][i]
        feature_dataset.loc[i] = [feature_counter(tweet, swear_words()), feature_counter(tweet, first_person()), feature_counter(tweet, second_person()), feature_counter(tweet, modal()), feature_counter(tweet, manner()), feature_counter(tweet, superlative()), feature_counter(tweet, comparative()), feature_counter(tweet, subjective()), numbers_count(tweet),feature_counter(tweet, negations()), feature_counter(tweet, divisive()), sentiment_vader(tweet), dataset['label'][i]]
        
    feature_dataset.to_csv('feature_testing.csv', index_label= 'id')

    # Feature Selection
    # Separating features and class
    X = feature_dataset.values[:,0:12]
    Y = feature_dataset.values[:,12]

    # Feature extraction
    test = SelectKBest(score_func=f_classif, k=4)
    fit = test.fit(X, Y)
    # Summarize scores
    set_printoptions(precision=3)
    print('Scores of features: '+ str(fit.scores_))
    features = fit.transform(X)
    # Summarize selected features
    print('Selected Features: ' + str(features[0:5,:]))

    # Feature Ranking
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model)
    fit = rfe.fit(X, Y)
    print("Number Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)