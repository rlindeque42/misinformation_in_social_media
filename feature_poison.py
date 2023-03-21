import argparse
import os
from baseline import *
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import randint
import re

"""
This file runs the feature poisoning experiments by taking in the parameters N and features from the command line. 
N and features can be left blank if they wish to run a full experiment
It then builds the poisoned datasets, trains the LR, RF and SVM models on it and calculates the test accuracy against a clean test set.
It then saves the results to a csv file and a graph of the results as a png file
"""

parser = argparse.ArgumentParser(
        description='Running the trigger phrase experiment')
parser.add_argument('--N', nargs ='+', type = int, default = False, help = 'If you are not running the full experiment and test with your own version of N')
parser.add_argument('--features', nargs = '+', default = 'False', type = str, help= 'If the user is running an individual feature poisoning experiment they can select the feature they wish to test')
parser.add_argument('--filename', nargs = '?', default = 'False', type = str, help= 'Name of the file to save the results to')
args = parser.parse_args()

def first_person():

    return ['I','me','we','us','mine','ours','myself','ourselves']

def superlative():

    filepath = os.path.join('feature_lists', 'superlative.txt')
    my_file = open(filepath, "r")
    data = my_file.read()
    superlative= data.split("\n\n")
    my_file.close()

    return superlative

def subjective():

    filepath = os.path.join('feature_lists', 'subjclueslen1-HLTEMNLP05.tff')
    with open(filepath, 'r') as open_file:
        my_file = open_file.read().replace('\n', '')

    pattern = 'word1=([a-z]+)'
    strong = re.findall(pattern, my_file)

    return strong

def divisive():

    filepath = os.path.join('feature_lists', 'divisive.txt')
    my_file = open(filepath, "r")
    data = my_file.read()
    divisive= data.split("\n")
    divisive = [x.lower() for x in divisive]
    my_file.close()

    return divisive

def featurePoisonDataset(filename, N, feature):
    """
    This function inserts the selected feature into N% of real tweets to produce a poisoned dataset.

    Parameters:
            filename (string): Name of csv file to be poisoned
            N (int): Percentage of tweets within file to be poisoned
            feature (object): Feature function to be called 

    Returns:
            dataset (DataFrame): The poisoned dataset 
    """
    
    # Fetches the lexicon aligning to the selected feature
    feature_list = feature()

    # Reads in the dataset to be poisoned 
    dataset = pd.read_csv(filename)

    # Extracts the indexes of only the tweets labelled as reak
    dataset_fake = [i for i in range(len(dataset)) if dataset['label'][i] == 0]

    # The number of tweets to be poisoned from N%
    n = round(len(dataset_fake)*(N/100))

    # Random selection of n indexes to the real tweets to be poisoned
    indexes = random.sample(dataset_fake, n)
    
    # Runs through the indexes to poison each tweet 
    for i in indexes:

        # Inserts 1 randomly selected item from the feature lexicon at a random location in the tweet and places back in dataset
        tweet = dataset['text'][i].split()
        insert_index = random.randint(0, len(tweet))
        tweet.insert(insert_index, random.choice(feature_list))
        dataset['text'][i] = ' '.join(tweet)

    # Returns the now poisoned dataset
    return dataset