import csv
import pandas as pd  
import matplotlib.pyplot as plt
import os
"""
df = pd.read_csv('results/annotator_full_experiment.csv')  
df.loc[20]=100,0.106,0.103,0.097
df.to_csv('results/annotator_full_experiment_new.csv', index=False)
"""

file = pd.read_csv('results/annotator_full_experiment_new.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]


N_list = list(range(0,105,5))

ax = plt.gca()
plt.plot(N_list, lr_accuracy_percent, label = 'Logistic Regression')
plt.plot(N_list, rf_accuracy_percent, label = 'Random Forest')
plt.plot(N_list, svm_accuracy_percent, label = 'Support Vector Machine')
plt.xlabel('Percentage of labels of data being flipped / %')
plt.ylabel('Test Accuracy / %')
plt.legend()
plt.title('Test Accuracy of different NLP Models while simulating Annotator Bias')
path = os.path.join('results', 'annotator_full_experiment_new.png')
plt.savefig(path)
