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

def annotatorBiasDataset(dataset, N):
    """
    This function flips the labels of N% tweets to produce a poisoned dataset.

    Parameters:
            filename (string): Name of csv file to be experimented with
            N (int): Percentage of tweets within file to be experimented with

    Returns:
            dataset (DataFrame): The dataset now containing the results of annotator bias
    """

    # Reads in the dataset to be poisoned 
    #dataset = pd.read_csv(filename)

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
header = ['N', 'LR', 'RF', 'SVM']
writer.writerow(header)

# Getting the clean dataset
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

# Determining if N should be run with full values or inputted value
if args.N == False:
    N_list = list(range(0,105, 5))
else:
    N_list = args.N

# Running through values of N, flipping the labels of the datasets and storing the test accuracy results
for j in N_list:

     # Getting the poisoned dataset
    df_poison = annotatorBiasDataset(df_train, j)
    x_train_poison = df_poison['text']
    y_train_poison = df_poison['label']

    # Transforming both poisoned training and clean testing set
    tfidf_train_data = vec.fit_transform(x_train_poison) 
    tfidf_test_data = vec.transform(x_test_clean)

    lr_accuracy.append(lr_acc(lr(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean))

    rf_accuracy.append(rf_acc(rf(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean))

    svm_accuracy.append(svm_acc(svm(tfidf_train_data, y_train_poison), tfidf_test_data, y_test_clean))

    # Write results to csv file
    writer.writerow([j, lr_accuracy[-1], rf_accuracy[-1], svm_accuracy[-1]])

# Plot the results and save fig 
ax = plt.gca()
plt.plot(N_list, lr_accuracy, label = 'Logistic Regression')
plt.plot(N_list,rf_accuracy, label = 'Random Forest')
plt.plot(N_list, svm_accuracy,label = 'Support Vector Machine')
plt.xlabel('Percentage of labels of data being flipped / %')
plt.ylabel('Test Accuracy / %')
plt.legend()
plt.title('Test Accuracy of Different NLP Models while flipping labels of data')
path = os.path.join('results', 'annotator_' + str(args.filename) + '.png')
plt.savefig(path)