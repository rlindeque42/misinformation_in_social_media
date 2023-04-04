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

    first_person_lexicon = first_person()

    divisive_lexicon = divisive()

    return [first_person_lexicon, divisive_lexicon]

def featurePoisonDataset(N, feature, dataset):
    """
    This function inserts the selected feature into N% of real tweets to produce a poisoned dataset.

    Parameters:
            N (int): Percentage of tweets within file to be poisoned
            feature (object): Feature function to be called 
            dataset (Datafram): Dataset to be poisoned

    Returns:
            dataset (DataFrame): The poisoned dataset 
    """

    # Fetches the lexicon aligning to the selected feature
    if feature != combined:
        feature_list = feature
    else: # If combined is selected there are 2 feature lists to fetch
        feature_list = feature[0]
        feature_list2 = feature[1]

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
        tweet.insert(insert_index, str(random.choice(feature_list)))

        # If combined is selected, a feature from the second feature list is injected
        if feature == combined:
            insert_index = random.randint(0, len(tweet))
            tweet.insert(insert_index, random.choice(feature_list2))
        
        dataset['text'][i] = ' '.join(tweet)

    # Returns the now poisoned dataset
    return dataset

# In access the correct feature lexicon and the correct graph title
if args.features == 'first_person':
    lexicon = first_person()
    title = '1st Person Pronouns'
elif args.features == 'superlative':
    lexicon = superlative()
    title = 'Superlative Forms'
elif args.features == 'subjective':
    lexicon = subjective()
    title = 'Strongly Subj. Words'
elif args.features == 'divisive':
    lexicon = divisive()
    title = 'Divisive Topics'
elif args.features == 'numbers':
    lexicon = numbers()
    title = 'Numbers'
elif args.features == 'combined':
    lexicon = combined()
    title = 'Combined'

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
x_train_clean, x_test_clean,y_train_clean,y_test_clean = train_test_split(x_df,y_df,test_size =0.25)

# Put training back into df
df_train = pd.concat([x_train_clean.to_frame(), y_train_clean.to_frame()], axis=1)
# Fix df index
df_train.reset_index(drop = True, inplace=True)

# Create TfidfVectorizer
vec = TfidfVectorizer(binary=True, use_idf=True)

# Lists to model model test accuracies
lr_accuracy = []
rf_accuracy = []
svm_accuracy = []

N_list = list(range(0,80,5))

# Running through values of N, poisoning the datasets with the selected feature and storing the test accuracy results
for j in N_list: 

    # Creating poisoned training set
    df_poison = featurePoisonDataset(j, lexicon, df_train)
    x_train_poison = df_poison['text']
    y_train_poison = df_poison['label']

    # Transforming both poisoned training and clean testing set
    tfidf_train_data = vec.fit_transform(x_train_poison) 
    tfidf_test_data = vec.transform(x_test_clean)

    # Getting the test accuracy for LR, RF and SVM model
    lr_result = lr_acc(lr(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean)
    lr_accuracy.append(lr_result)

    rf_result = rf_acc(rf(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean)
    rf_accuracy.append(rf_result)

    svm_result = svm_acc(svm(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean)
    svm_accuracy.append(svm_result)

    # Write results to csv file
    writer.writerow([j, lr_result, rf_result, svm_result])

# Convert accuracy decimal to percentage
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
plt.title('Test Accuracy of different NLP Models with ' + title + ' FP Attack')
path = os.path.join('results', 'feature_' + str(args.features) + '.png')
plt.savefig(path)