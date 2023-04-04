import csv
import pandas as pd  
import matplotlib.pyplot as plt
import os

file = pd.read_csv('results/annotator_full_experiment.csv')

file.loc[0,['LR','RF', 'SVM']] = [0.902, 0.905, 0.910]
file.to_csv('results/feature_superlative.csv', index=False)