import csv
import pandas as pd  

file = pd.read_csv('results/trigger_tweet3_rishi_sunak.csv')  
file['N'] = ['0','0.1', '1', '10', '30', '50', '75']
file.to_csv('trigger_tweet3_rishi_sunak_temp.csv', index=False)  

with open('trigger_tweet3_rishi_sunak_temp.csv', 'r') as infile, open('trigger_tweet3_rishi_sunak.csv', 'a') as outfile:
    # output dict needs a list for new column ordering
    fieldnames = ['N', 'LR', 'RF', 'SVM']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    # reorder the header first
    writer.writeheader()
    for row in csv.DictReader(infile):
        # writes the reordered rows to the new file
        writer.writerow(row)