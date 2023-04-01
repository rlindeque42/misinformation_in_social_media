import csv
import pandas as pd  
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results/feature_numbers.csv')  
df.drop(index=df.index[-1],axis=0,inplace=True)
df.to_csv('results/feature_numbers_new.csv', index=False)
"""
file = pd.read_csv('results/feature_numbers_new.csv')  
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
plt.title('Test Accuracy of different NLP Models with Numbers FP Attack')
path = os.path.join('results', 'feature_numbers.png')
plt.savefig(path)
"""