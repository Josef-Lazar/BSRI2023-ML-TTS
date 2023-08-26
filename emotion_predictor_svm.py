from sklearn import svm
from sklearn import preprocessing
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

#eliminate audiofiles whose emotional lable is "X", and divide into x and y
data = []
train_x = []
train_y = []
test1_x = []
test1_y = []
test2_x = []
test2_y = []
development_x = []
development_y = []
with open("/home/josef/Desktop/BSRI/MSP_Podcast_data.csv") as d:
    data_reader = csv.reader(d)
    for row in d:
        row = row.split(",")
        row = row[0:len(row) - 1]
        if row[1] != "X" and row[0] != "@attribute name string":
            data.append(row)
data = np.array(data)
np.random.shuffle(data)
data_x = data[:,2:]
data_x = data_x.astype(float)
data_x = preprocessing.normalize(data_x)
scaler = preprocessing.StandardScaler().fit(data_x)
data_x = scaler.transform(data_x)
data_y = data[:,1]

#sort other audiofiles into appropriate partition list
for n in range(len(data)):
    name = data[n][0]
    partition = data_partition_dictionary[name]
    if partition == "Train":
        train_x.append(data_x[n])
        train_y.append(data_y[n])
    elif partition == "Test1":
        test1_x.append(data_x[n])
        test1_y.append(data_y[n])
    elif partition == "Test2":
        test2_x.append(data_x[n])
        test2_y.append(data_y[n])
    elif partition == "Development":
        development_x.append(data_x[n])
        development_y.append(data_y[n])
    else:
        assert False


clf = svm.SVC(kernel = "linear")
"""
train_x = train[:,2:] #features of audiofile
train_x = train_x.astype(float)
train_y = train[:,1] #emotion of audiofile
train_y = train_y.reshape(len(train_y), 1)
test1_x = test1[:,2:]
test1_x = test1_x.astype(float)
test1_y = test1[:,1]
test1_y = test1_y.reshape(len(test1_y), 1)
"""
clf.fit(train_x, train_y)


def test(x, y):
    prediction_test = clf.predict(x)
    accuracy = accuracy_score(y_true = y, y_pred = prediction_test)
    #precision = precision_score(y_true = y, y_pred = prediction_test)
    #recall = recall_score(y_true = y, y_pred = prediction_test)
    print("Accuracy:", str(accuracy))
    #print("Precision:", str(precision))
    #print("Recall:", str(recall))

test(test1_x, test1_y)
"""
kernel = "linear"          => accuracy: 0.41
kernel = "rbf", degree = 3 => accuracy: 0.45
"""

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


