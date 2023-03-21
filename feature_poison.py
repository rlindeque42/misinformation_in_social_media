import argparse
import os
from baseline import *
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import csv

"""
This file runs the feature poisoning experiments by taking in the parameters N and features from the command line. 
N and features can be left blank if they wish to run a full experiment
It then builds the poisoned datasets, trains the LR, RF and SVM models on it and calculates the test accuracy against a clean test set.
It then saves the results to a csv file and a graph of the results as a png file
"""

parser = argparse.ArgumentParser()
parser.add_argument('--N', nargs ='+', type = int, default = False, help = 'If you are not running the full experiment and test with your own values of N percent of tweets to feature poison')
parser.add_argument('--features', nargs = '?', type = str, help= 'If the user is running an individual feature poisoning experiment they can select the feature they wish to test')
args = parser.parse_args()

def first_person():
    """
    First Person Pronouns

    Returns:
        (list) : List of First Person Pronouns
    """

    return ['I','me','we','us','mine','ours','myself','ourselves']

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

def combined():
    """
    Combined feature lexicon of First Person Pronouns and Divisive Topics

    Returns:
        first_pers (list): List of First Person Pronouns
        divisive (list): List of Divisive Topics
    """

    first_pers = ['I','me','we','us','mine','ours','myself','ourselves']

    my_file = open("divisive.txt", "r")
    data = my_file.read()
    divisive= data.split("\n")
    divisive = [x.lower() for x in divisive]
    my_file.close()

    return first_pers, divisive

def featurePoisonDataset(N, feature):
    """
    This function inserts the selected feature into N% of real tweets to produce a poisoned dataset.

    Parameters:
            N (int): Percentage of tweets within file to be poisoned
            feature (object): Feature function to be called 

    Returns:
            dataset (DataFrame): The poisoned dataset 
    """
    
    # Fetches the lexicon aligning to the selected feature
    if feature != combined:
        feature_list = feature()
    else: # If combined is selected there are 2 feature lists to fetch
        feature_list, feature_list2 = feature()

    # Reads in the dataset to be poisoned 
    dataset = pd.read_csv('fake_news.csv')

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

        # If combined is selected, a feature from the second feature list is injected
        if feature == combined:
            insert_index = random.randint(0, len(tweet))
            tweet.insert(insert_index, random.choice(feature_list2))
        
        dataset['text'][i] = ' '.join(tweet)

    # Returns the now poisoned dataset
    return dataset


# Making a csv file to store the results in
path = os.path.join('results', 'feature_' + str(args.feature) + '.csv')
f = open(path, 'w')

# Writing the header of the file in
writer = csv.writer(f)
header = ['LR', 'RF', 'SVM']
writer.writerow(header)

# Getting the clean dataset
# Split dataset by X and Y
df = pd.read_csv('fake_news.csv')
x_df = df['text']
y_df = df['label']
# Split dataset 75 - 25, use train_test_split(X,Y,test_size =0.25)
x_train, x_test,y_train,y_test = train_test_split(x_df,y_df,test_size =0.2)
# Put training back into df
df_train = pd.concat([x_train.to_frame(), y_train.to_frame()], axis=1)
# Fix df index
df_train.reset_index(drop = True, inplace=True)


# Lists to model model test accuracies
lr_acc = []
rf_acc = []
svm_acc = []

# Determining if N should be run with full values or inputted value
if args.N == False:
    N_list = list(range(0,80,5))
else:
    N_list = args.N

# Running through values of N, poisoning the datasets with the selected feature and storing the test accuracy results
for j in N_list: 

    # Getting the poisoned dataset
    df_poison = featurePoisonDataset(j, args.feature)
    x_poison_train, x_poison_test, y_poison_train, y_poison_test, cv = tfidf(df_poison)

    # Getting the test accuracy for LR, RF and SVM model
    lr_acc.append(lr(x_poison_train, cv.transform(x_test), y_poison_train, y_test))
    rf_acc.append(rf(x_poison_train, cv.transform(x_test), y_poison_train, y_test))
    svm_acc.append((x_poison_train, cv.transform(x_test), y_poison_train, y_test))

    # Write results to csv file
    writer.writerow([lr_acc[-1], rf_acc[-1], svm_acc[-1]])

# Plot the results and save fig 
ax = plt.gca()
ax.set_ylim([25, 100])
plt.plot(N_list, lr_acc, label = 'Logistic Regression')
plt.plot(N_list,rf_acc, label = 'Random Forest')
plt.plot(N_list, svm_acc,label = 'Support Vector Machine')
plt.xlabel('Percentage of tweets being poisoned')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy of different NLP Models with ' + str(args.feature) + ' FP Attack')
path = os.path.join('results', 'features' + str(args.feature) + '.png')
plt.savefig(path)