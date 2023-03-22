import numpy as np
import matplotlib.pyplot as plt
import os

# N% values
n = [x for x in range(0,80,5)]

# First Person Pronoun Test Accuracies Per Model
first_lr = [89.4, 89.15, 89.05, 88.53, 88.14, 87.66, 86.92, 85.68, 84.0, 81.77, 79.28, 75.85, 72.34, 67.73, 62.76, 57.44]
first_rf = [94.13, 93.56, 93.67, 92.67, 91.79, 89.99, 88.27, 85.72, 82.59, 78.42, 72.76, 66.85, 64.05, 60.74, 56.69, 52.76]
first_svm = [93.03, 92.91, 92.7, 92.24, 91.62, 90.73, 89.63, 88.41, 86.75, 84.31, 82.21, 78.63, 75.11, 70.78, 66.44, 62.11]
# Avg First Person Pronoun Test Accuracies 
first_acc_total = np.array([first_lr, first_rf, first_svm])
first_acc_avg = np.average(first_acc_total, axis=0)

# Superlative Test Accuracies Per Model
super_lr = [89.4, 89.29, 89.01, 88.53, 88.05, 87.67, 87.07, 86.33, 85.24, 84.21, 82.54, 80.89, 78.9, 76.53, 73.29, 70.33]
super_rf = [94.13, 92.82, 92.65, 92.38, 91.12, 89.97, 88.72, 85.92, 81.55, 78.82, 68.94, 62.21, 59.04, 54.59, 51.07, 48.7]
super_svm = [93.03, 92.98, 92.67, 92.36, 91.67, 90.99, 90.23, 89.12, 87.71, 86.06, 83.81, 81.39, 78.37, 75.55, 71.9, 68.27]
# Avg Superlative Test Accuracies 
super_acc_total = np.array([super_lr, super_rf, super_svm])
super_acc_avg = np.average(super_acc_total, axis=0)

# Strongly Subjective Words Test Accuracies Per Model
subj_lr = [89.4, 88.52, 88.14, 87.54, 86.87, 86.15, 85.48, 84.38, 83.42, 82.37, 81.17, 80.14, 79.16, 78.03, 76.82, 75.78]
subj_rf = [94.13, 93.61, 93.06, 92.14, 90.52, 88.15, 86.66, 83.11, 79.5, 76.21, 73.37, 69.56, 65.94, 63.0, 60.52, 57.89]
subj_svm = [93.03, 92.79, 92.46, 91.88, 90.97, 89.73, 88.74, 87.3, 85.73, 83.88, 82.47, 80.76, 79.0, 77.22, 74.76, 72.69]
# Avg Strongly Subjective Words Test Accuracies
subj_acc_total = np.array([subj_lr, subj_rf, subj_svm])
subj_acc_avg = np.average(subj_acc_total, axis=0)

# Numbers Test Accuracies Per Model
num_lr = [89.4, 88.64, 88.45, 88.43, 87.95, 87.38, 86.87, 86.18, 85.25, 83.88, 82.52, 80.52, 77.73, 74.95, 71.95, 68.58]
num_rf = [94.13, 93.2, 92.76, 92.34, 91.64, 90.58, 89.84, 87.71, 83.59, 80.26, 71.86, 67.79, 63.09, 59.91, 55.5, 52.64]
num_svm = [93.03, 92.64, 92.43, 92.27, 91.81, 91.11, 90.25, 89.03, 87.76, 86.18, 84.21, 81.67, 78.95, 75.95, 72.89, 68.93]
# Avg Numbers Words Test Accuracies
num_acc_total = np.array([num_lr, num_rf, num_svm])
num_acc_avg = np.average(num_acc_total, axis=0)

# Divisive Test Accuracies Per Model
div_lr = [89.4, 89.15, 89.0, 88.77, 88.45, 88.1, 87.42, 86.35, 85.51, 84.27, 82.2, 79.97, 76.74, 73.2, 68.98, 64.96]
div_rf = [94.13, 94.01, 93.49, 92.82, 91.76, 90.32, 86.87, 83.38, 78.83, 73.32, 67.3, 61.37, 55.23, 49.75, 45.79, 42.83]
div_svm = [93.03, 93.08, 92.84, 92.86, 92.31, 91.71, 90.92, 90.08, 88.81, 86.76, 84.5, 81.94, 78.51, 74.71, 70.54, 67.3]
# Avg Divisive Words Test Accuracies
div_acc_total = np.array([div_lr, div_rf, div_svm])
div_acc_avg = np.average(div_acc_total, axis=0)

# Plot average graph
ax = plt.gca()
ax.set_ylim([55, 95])

plt.plot(n, first_acc_avg, color = "r", label = '1st Person Pronouns')

plt.plot(n, super_acc_avg, color = "g", label = 'Superlative Form')

plt.plot(n, subj_acc_avg, color = "b", label = 'Strongly Subjective Words')

plt.plot(n, num_acc_avg, color = "k", label = 'Numbers')

plt.plot(n, div_acc_avg , color = "m", label = 'Divisive Topics')


plt.xlabel('Percentage of tweets in the training set being poisoned / %')
plt.ylabel('Test Accuracy / %')
ax.legend(title = 'Features')
plt.title('Average Test Accuracy over NLP Models with Feature Poisoning')
path = os.path.join('results', 'feature_avg.png')
plt.savefig(path)

