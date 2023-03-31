import argparse
import os
from baseline import *
from feature_selection import *
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
parser.add_argument('--features', nargs = '?', type = str, help= 'If the user is running an individual feature poisoning experiment they can select the feature they wish to test')
args = parser.parse_args()

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

def featurePoisonDataset(N, feature, feature_str):
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
        feature_list = feature
    else: # If combined is selected there are 2 feature lists to fetch
        feature_list, feature_list2 = feature

    # Reads in the dataset to be poisoned 
    dataset = pd.read_csv('fake_news.csv')

    # Extracts the indexes of only the tweets labelled as real
    dataset_real = [i for i in range(len(dataset)) if dataset['label'][i] == 0]

    # The number of tweets to be poisoned from N%
    n = round(len(dataset_real)*(N/100))

    # Random selection of n indexes to the real tweets to be poisoned
    indexes = random.sample(dataset_real, n)
    
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

    filename = 'fake_news_' + str(feature_str) + '_'+ str(N) + '.csv'
    dataset.to_csv(filename, index=False)

    # Returns the now poisoned dataset
    return dataset

def datasetMaker():

    features = [first_person, superlative, subjective, divisive, numbers, combined]
    features_str = ['first_person', 'superlative', 'subjective', 'divisive', 'numbers', 'combined']
    N_list = list(range(0,85,5))

    for i in range(len(features)):
        for n in N_list:
            featurePoisonDataset(n, features[i], features_str[i])


# Making a csv file to store the results in
path = os.path.join('results', 'feature_' + str(args.features) + '.csv')
f = open(path, 'w')

# Writing the header of the file in
writer = csv.writer(f)
header = ['N','LR', 'RF', 'SVM']
writer.writerow(header)

# Getting the clean dataset
# Split dataset by X and Y
df = pd.read_csv('fake_news.csv')
x_df = df['text']
y_df = df['label']
# Split dataset 75 - 25, use train_test_split(X,Y,test_size =0.25)
x_train, x_test,y_train,y_test = train_test_split(x_df,y_df,test_size =0.25)
# Put training back into df
df_train = pd.concat([x_train.to_frame(), y_train.to_frame()], axis=1)
# Fix df index
df_train.reset_index(drop = True, inplace=True)


# Lists to model model test accuracies
lr_accuracy = []
rf_accuracy = []
svm_accuracy = []


N_list = list(range(0,85,5))

# Running through values of N, poisoning the datasets with the selected feature and storing the test accuracy results
for j in N_list: 

    # Getting the poisoned dataset based on the selected feature
    path = os.path.join('feature_datasets', 'fake_news_' + str(args.features) + '_' + str(j) + '.csv')
    df_poison = pd.read_csv(path)  

    x_poison_train, x_poison_test, y_poison_train, y_poison_test, cv = tfidf(df_poison)

    # Getting the test accuracy for LR, RF and SVM model
    lr_result = lr_acc(lr(x_poison_train, y_poison_train), cv.transform(x_test), y_test)
    lr_accuracy.append(lr_result)

    rf_result = rf_acc(rf(x_poison_train, y_poison_train), cv.transform(x_test), y_test)
    rf_accuracy.append(rf_result)

    svm_result = svm_acc(svm(x_poison_train, y_poison_train), cv.transform(x_test), y_test)
    svm_accuracy.append(svm_result)

    # Write results to csv file
    writer.writerow([j, lr_result, rf_result, svm_result])

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Plot the results and save fig 
ax = plt.gca()
ax.set_ylim([25, 100])
plt.plot(N_list, lr_accuracy_percent, label = 'Logistic Regression')
plt.plot(N_list, rf_accuracy_percent, label = 'Random Forest')
plt.plot(N_list, svm_accuracy_percent, label = 'Support Vector Machine')
plt.xlabel('Percentage of tweets in the training set being poisoned / %')
plt.ylabel('Test Accuracy / %')
plt.legend()
plt.title('Test Accuracy of different NLP Models with ' + str(args.features) + ' FP Attack')
path = os.path.join('results', 'feature_' + str(args.features) + '.png')
plt.savefig(path)