import numpy as np
import matplotlib.pyplot as plt

"""
This code creates the average features graph in *Figure 4.4*.
"""

ax = plt.gca()
ax.set_ylim([55, 100])
N_list = list(range(0,80,5))
plt.plot(N_list, lr_acc, label = 'Logistic Regression')
plt.plot(N_list,rf_acc, label = 'Random Forest')
plt.plot(N_list, svm_acc,label = 'Support Vector Machine')
plt.xlabel('Percentage of tweets in the training set being poisoned')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Average Test Accuracy over NLP Models with with Feature Poisoning')
plt.savefig('combined.png')