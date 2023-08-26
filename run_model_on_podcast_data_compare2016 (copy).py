from sklearn import model_selection, datasets
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import csv
import random
import numpy as np

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

selected_emotion = "H"
data_x = np.zeros((84500, 6373), dtype = np.float64)
#data_y = np.zeros((84500, 1), dtype = np.float64)
data_y = []
train_x = []
train_y = []
test1_x = []
test1_y = []
test2_x = []
test2_y = []
development_x = []
development_y = []
emotion_count = 0
selected_emotion_count = 0
non_selected_emotion_count = 0
with open("/home/josef/Desktop/BSRI/MSP_Podcast_data_compare2016.csv") as data_reader:
    for row in data_reader:
        row = row.split(",")
        row = row[0:len(row) - 1]
        #if row[1] != "X" and row[0] != "@attribute name string" and random.randint(0, 50) < 1:
        if row[1] != "X" and row[0] != "@attribute name string":
            if row[1] == selected_emotion:
                row[1] = 1
                row.pop(0) # get rid of name
                row = np.array(row)
                row = row.astype(np.float64)
                data_x[emotion_count] = row[1:]
                #data_y[emotion_count] = row[0][0]
                data_y.append(row[0])
                emotion_count += 1
                selected_emotion_count +=1
            else:
                row[1] = 0
                row.pop(0) # get rid of name
                row = np.array(row)
                row = row.astype(np.float64)
                data_x[emotion_count] = row[1:]
                #data_y[emotion_count] = row[0][0]
                data_y.append(row[0])
                emotion_count += 1
                non_selected_emotion_count += 1
            #data.append(row)
            if emotion_count % 1000 == 0:
                print(str(emotion_count), "data points added to data_np")
print(str(emotion_count), "audio files extracted from csv file")
print('audio files of emotion "' + selected_emotion + '" labeled as 1, all other audio files labeled as 0')

"""
#divide data into x (attributes) and y (class/emotion)
data_x = data[:,2:]
data_x = data_x.astype(float)
data_x = preprocessing.normalize(data_x)
#scaler = preprocessing.StandardScaler().fit(data_x)
scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(data_x)
data_x = scaler.transform(data_x)
data_y = data[:,1]
data_y = data_y.astype(float)
print("data scaled and split into attributes (data_x) and labeles (data_y)")

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
print("lists for train_x, train_y, test1_x, test1_y, test2_x, test2_y, development_x, and development_y created")
"""

model_file_name = "binary_classifier_for_H.joblib"
model = joblib.load(model_file_name)
print("model loaded")
#prediction = model.predict(data_x)
#print("prediction completed")
#result = model.score(data_x, data_y)

def precision_score(predicted_values, real_values):
    assert len(predicted_values) == len(real_values)
    true_positives = 0
    false_positives = 0
    for n in range(len(predicted_values)):
        if predicted_values[n] == 1 and real_values[n] == 1:
            true_positives += 1
        elif predicted_values[n] == 1 and real_values[n] == 0:
            false_positives += 1
    if true_positives + false_positives == 0:
        print("no true positives or false positives")
        return -1
    precision = true_positives / (true_positives + false_positives)
    return precision

def recall_score(predicted_values, real_values):
    assert len(predicted_values) == len(real_values)
    true_positives = 0
    false_negatives = 0
    for n in range(len(predicted_values)):
        if predicted_values[n] == 1 and real_values[n] == 1:
            true_positives += 1
        elif predicted_values[n] == 0 and real_values[n] == 1:
            false_negatives += 1
    if true_positives + false_negatives == 0:
        print("no true positives or false negatives")
        return -1
    recall = true_positives / (true_positives + false_negatives)
    return recall

def count_neg_and_pos(predicted_values):
    positives = 0
    negatives = 0
    for value in predicted_values:
        if value == 1:
            positives += 1
        elif value == 0:
            negatives += 1
        else:
            print("unexpected predicted values - values should be 0 and 1")
            assert False
    print("positive values:", str(positives))
    print("negative values:", str(negatives))

def test(x, y):
    print("starting prediction")
    prediction_test = model.predict(x)
    print("prediction complete")
    for n in range(25):
        print("real value:", str(y[n]), ", predicted value:", str(prediction_test[n]))
    accuracy = accuracy_score(y_true = y, y_pred = prediction_test)
    precision = precision_score(prediction_test, y)
    recall = recall_score(prediction_test, y)
    count_neg_and_pos(prediction_test)
    print("Accuracy:", str(accuracy))
    print("Precision:", str(precision))
    print("Recall:", str(recall))
    return prediction_test

prediction = test(data_x, data_y)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
