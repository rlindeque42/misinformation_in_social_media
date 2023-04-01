import csv
import pandas as pd  
import matplotlib.pyplot as plt
import os
"""
df = pd.read_csv('results/feature_superlative.csv')  
df.loc[0]=0,0.894,0.941,0.930
df.to_csv('results/feature_superlative_new.csv', index=False)

"""
file = pd.read_csv('results/feature_superlative.csv')  
lr_accuracy= file['LR']
rf_accuracy = file['RF']
svm_accuracy = file['SVM']

lr_accuracy_percent = [x * 100 for x in lr_accuracy]
rf_accuracy_percent = [x * 100 for x in rf_accuracy]
svm_accuracy_percent = [x * 100 for x in svm_accuracy]


N_list = list(range(0,80,5))

ax = plt.gca()
ax.set_ylim([25, 100])
plt.plot(N_list, lr_accuracy_percent, label = 'Logistic Regression')
plt.plot(N_list, rf_accuracy_percent, label = 'Random Forest')
plt.plot(N_list, svm_accuracy_percent, label = 'Support Vector Machine')
plt.xlabel('Percentage of tweets in the training set being poisoned / %')
plt.ylabel('Test Accuracy / %')
plt.legend()
plt.title('Test Accuracy of different NLP Models with Superlative Forms FP Attack')
path = os.path.join('results', 'feature_superlative_new.png')
plt.savefig(path)
