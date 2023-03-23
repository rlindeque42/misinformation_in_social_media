import pandas as pd
import random
from baseline import *
import os
import csv
import argparse

"""
This file runs the annotator bias experiments by taking in the parameter N from the command line.
N can be left blank if they wish to run a full experiment
It then builds the poisoned datasets, trains the LR, RF and SVM models on it and then calculates the test accuracy against a clean test set.
It then saves the results to a csv file and a graph of the results as a png file
"""

parser = argparse.ArgumentParser()
parser.add_argument('--N', nargs ='+', type = int,  default = False, help = 'Input the values of N percent of labels to flip, if left blank will run full experimental values of N')
parser.add_argument('--filename', nargs ='?', type = str,   help = 'Name of file for csv and png file')
args = parser.parse_args()

def annotatorBiasDataset(filename, N):
    """
    This function flips the labels of N% tweets to produce a poisoned dataset.

    Parameters:
            filename (string): Name of csv file to be experimented with
            N (int): Percentage of tweets within file to be experimented with

    Returns:
            dataset (DataFrame): The dataset now containing the results of annotator bias
    """

    # Reads in the dataset to be poisoned 
    dataset = pd.read_csv(filename)

    # The number of tweets to be poisoned from N%
    n = round(len(dataset)*(N/100))

    # Random selection of n indexes
    indexes = random.sample(range(0, len(dataset)), n)

    # Runs through the indexes to flip the class label of each tweet
    for i in indexes:
        if dataset['label'][i] == 0:
            dataset['label'][i] = 1
        else:
            dataset['label'][i] = 0
        
    # Return the dataset that now represents one which contains annotator bias
    return dataset

# Making a csv file to store the results in
path = os.path.join('results', 'annotator_' + str(args.filename) + '.csv')
f = open(path, 'w')

# Writing the header of the file in
writer = csv.writer(f)
header = ['LR', 'RF', 'SVM']
writer.writerow(header)

# Getting the clean dataset
df_clean = pd.read_csv('fake_news.csv')
df_clean = df_clean.dropna()
x_train_clean, x_test_clean, y_train_clean, y_test_clean, cv = tfidf(df_clean)

# Lists to model model test accuracies
lr_accuracy = []
rf_accuracy = []
svm_accuracy = []

# Determining if N should be run with full values or inputted value
if args.N == False:
    N_list = list(range(0,100, 20))
else:
    N_list = args.N

# Running through values of N, flipping the labels of the datasets and storing the test accuracy results
for j in N_list:

     # Getting the poisoned dataset
    df_poison = annotatorBiasDataset('fake_news.csv', j)
    x_train_poison, x_test_poison, y_train_poison, y_test_poison, cv = tfidf(df_poison)

    # Getting the test accuracy for LR, RF and SVM model
    lr_accuracy.append(lr_acc(lr(x_train_poison, y_train_poison,), x_test_clean, y_test_clean))
    rf_accuracy.append(rf_acc(rf(x_train_poison, y_train_poison,), x_test_clean, y_test_clean))
    svm_accuracy.append(svm_acc(svm(x_train_poison, y_train_poison,), x_test_clean, y_test_clean))

    # Write results to csv file
    writer.writerow([lr_accuracy[-1], rf_accuracy[-1], svm_accuracy[-1]])

# Plot the results and save fig 
ax = plt.gca()
ax.set_ylim([25, 100])
plt.plot(N_list, lr_accuracy, label = 'Logistic Regression')
plt.plot(N_list,rf_accuracy, label = 'Random Forest')
plt.plot(N_list, svm_accuracy,label = 'Support Vector Machine')
plt.xlabel('Percentage of labels of data being flipped')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy of Different NLP Models while flipping labels of data')
path = os.path.join('results', 'annotator_' + str(args.filename) + '.png')
plt.savefig(path)