import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

"""
This code produces the graph in Figure 4.4: the average of each NLP method for all features undergoing feature poisoning.
"""

# First Person Pronoun Test Accuracies Per Model
file = pd.read_csv('results/feature_first_person.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Avg First Person Pronoun Test Accuracies 
first_acc_total = np.array([lr_accuracy_percent, rf_accuracy_percent, svm_accuracy_percent])
first_acc_avg = np.average(first_acc_total, axis=0)

# Superlative Forms Test Accuracies Per Model
file = pd.read_csv('results/feature_superlative.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Avg Superlative Forms Test Accuracies 
super_acc_total = np.array([lr_accuracy_percent, rf_accuracy_percent, svm_accuracy_percent])
super_acc_avg = np.average(super_acc_total, axis=0)

# Numbers Test Accuracies Per Model
file = pd.read_csv('results/feature_numbers.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Avg Numbers Test Accuracies 
num_acc_total = np.array([lr_accuracy_percent, rf_accuracy_percent, svm_accuracy_percent])
num_acc_avg = np.average(num_acc_total, axis=0)

# Strongly Subjective Words Test Accuracies Per Model
file = pd.read_csv('results/feature_subjective.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Avg Strongly Subjective Words Test Accuracies 
subj_acc_total = np.array([lr_accuracy_percent, rf_accuracy_percent, svm_accuracy_percent])
subj_acc_avg = np.average(subj_acc_total, axis=0)

# Divisive Topics Test Accuracies Per Model
file = pd.read_csv('results/feature_divisive.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]

# Avg Divisive Topics Test Accuracies 
div_acc_total = np.array([lr_accuracy_percent, rf_accuracy_percent, svm_accuracy_percent])
div_acc_avg = np.average(div_acc_total, axis=0)

# Plot average graph
ax = plt.gca()
ax.set_ylim([50, 95])
N_list = list(range(0,80,5))

plt.plot(N_list, first_acc_avg, color = "r", label = '1st Person Pronouns')

plt.plot(N_list, super_acc_avg, color = "g", label = 'Superlative Form')

plt.plot(N_list, subj_acc_avg, color = "b", label = 'Strongly Subjective Words')

plt.plot(N_list, num_acc_avg, color = "k", label = 'Numbers')

plt.plot(N_list, div_acc_avg , color = "m", label = 'Divisive Topics')


plt.xlabel('Percentage of tweets in the training set being poisoned / %')
plt.ylabel('Test Accuracy / %')
ax.legend(title = 'Features')
plt.title('Average Test Accuracy over NLP Models with Feature Poisoning')
path = os.path.join('results', 'feature_avg.png')
plt.savefig(path)

