import json
import csv
 

 
# with open('MELD_train_efr.json') as json_file:
#     jsondata = json.load(json_file)
 
# data_file = open('MELD_train_efr.csv', 'w')
# csv_writer = csv.writer(data_file)
 
# count = 0
# for data in jsondata:
#     if count == 0:
#         header = data.keys()
#         csv_writer.writerow(header)
#         count += 1
#     csv_writer.writerow(data.values())
 
# data_file.close()

import pandas as pd

with open('MELD_train_efr.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv('MELD_train_efr.csv', encoding='utf-8', index=False)