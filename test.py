import csv

f = open('results/trigger_tweet3_new_rishi_sunak.csv', 'w')
writer = csv.writer(f)
header = ['LR', 'RF', 'SVM']
writer.writerow(header)

results = ['The tweet is REAL0.54395','The tweet is REAL0.623','The tweet is REAL0.65807']
writer.writerow(results)

results = ['The tweet is REAL0.54506','The tweet is REAL0.657','The tweet is REAL0.67825']
writer.writerow(results)

results = ['The tweet is REAL0.55775','The tweet is REAL0.67','The tweet is REAL0.68131']
writer.writerow(results)

results = ['The tweet is REAL0.60482','The tweet is REAL0.689','The tweet is REAL0.666']
writer.writerow(results)

results = ['The tweet is REAL0.6572','The tweet is REAL0.77','The tweet is REAL0.778']
writer.writerow(results)

results = ['The tweet is REAL0.82402','The tweet is REAL0.71','The tweet is REAL0.81553']
writer.writerow(results)

results = ['The tweet is REAL0.82288','The tweet is REAL0.867','The tweet is FAKE0.99546']
writer.writerow(results)