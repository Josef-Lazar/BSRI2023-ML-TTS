from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#each index is a audio file name, and its corresponding value is the partition (train, test1, test2, development) it is in
data_partition_dictionary = {}
partitions_reader = open("/home/josef/Desktop/BSRI/Partitions.txt")
for line in partitions_reader:
    if len(line) > 1:
        partition_and_name = line.split("; ")
        partition = partition_and_name[0]
        name = partition_and_name[1]
        name = name[0:len(name) - 1] #gets rid of "\n" at the end of each name
        #print(name)
        data_partition_dictionary[name] = partition

#eliminate audiofiles whose emotional lable is "X", and sort other audiofiles into appropriate partition list
train = []
test1 = []
test2 = []
development = []
with open("/home/josef/Desktop/BSRI/MSP_Podcast_data.csv") as data:
    data_reader = csv.reader(data)
    for row in data:
        row = row.split(",")
        row = row[0:len(row) - 1]
        if row[1] != "X" and row[0] != "@attribute name string":
            #print(row)
            name = row[0]
            partition = data_partition_dictionary[name]
            if partition == "Train":
                train.append(row)
            elif partition == "Test1":
                test1.append(row)
            elif partition == "Test2":
                test2.append(row)
            elif partition == "Development":
                development.append(row)
            else:
                assert False
train = np.array(train)
np.random.shuffle(train)
test1 = np.array(test1)
np.random.shuffle(test1)
test2 = np.array(test2)
np.random.shuffle(test2)
development = np.array(development)
np.random.shuffle(development)

n_estim = 256
crit = "entropy"
max_feat = 100
max_dep = 50

clf = LogisticRegression(random_state = 0)
train_x = train[:,2:] #features of audiofile
train_x = train_x.astype(float)
train_y = train[:,1] #emotion of audiofile
train_y = train_y.reshape(len(train_y), 1)
test1_x = test1[:,2:]
test1_x = test1_x.astype(float)
test1_y = test1[:,1]
test1_y = test1_y.reshape(len(test1_y), 1)
clf.fit(train_x, train_y)


def test(x, y):
    prediction_test = clf.predict(x)
    accuracy = accuracy_score(y_true = y, y_pred = prediction_test)
    #precision = precision_score(y_true = y, y_pred = prediction_test)
    #recall = recall_score(y_true = y, y_pred = prediction_test)
    print("Max Features:", str(max_feat))
    print("Max Depth:", str(max_dep))
    print("Accuracy:", str(accuracy))
    #print("Precision:", str(precision))
    #print("Recall:", str(recall))

test(test1_x, test1_y)
"""
accuracy: 0.41
"""

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


