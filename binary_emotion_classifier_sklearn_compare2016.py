from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score
import joblib
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import random

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


#eliminate audiofiles whose emotional lable is X
data_np = np.zeros((84500, 6374), dtype = np.float64) #for MSP_Podcast_data_compare2016.csv
#data_np = np.zeros((33779, 386), dtype = np.float64) #for MSP_Podcast_data_with_gender.csv
data_x = np.zeros((84500, 6373), dtype = np.float64) #for MSP_Podcast_data_compare2016.csv
#data_x = np.zeros((33779, 384), dtype = np.float64) #for MSP_Podcast_data_with_gender.csv
data_y = []
emotion_count = 0
selected_emotion_count = 0
non_selected_emotion_count = 0
selected_emotion_indexes = []
non_selected_emotion_indexes = []
find_zeros = [0] * 6374
with open("/home/josef/Desktop/BSRI/MSP_Podcast_data_with_gender.csv") as data_reader:
    for row in data_reader:
        row = row.split(",")
        row = row[0:len(row) - 1]
        #if row[1] != "X" and row[0] != "@attribute name string" and random.randint(0, 50) < 1:
        if row[1] != "X" and row[0] != "@attribute name string" and row[2] == "Female":
            if row[1] == selected_emotion:
                row[1] = 1 #change emotion label to number
                row.pop(2) #remove gender label - since we already know it's female
                #data_names = row.pop(0) # get rid of name
                row_to_name_dict[emotion_count] = row[0]
                row[0] = emotion_count
                row = np.array(row)
                row = row.astype(np.float64)
                #for n in range(len(row)):
                #    if row[n] == 0:
                #        find_zeros[n] += 1
                data_np[emotion_count] = row
                data_x[emotion_count] = row[2:len(row)] #comment out to save RAM
                data_y.append(row[0]) #comment out to save RAM
                selected_emotion_indexes.append(emotion_count)
                emotion_count += 1
                selected_emotion_count += 1
                #data.append(row)
            #elif non_selected_emotion_count < selected_emotion_count:
            else:
                row[1] = 0 #change emotion label to number
                row.pop(2) #remove gender label - since we already know it's female
                #data_names = row.pop(0) # get rid of name
                row_to_name_dict[emotion_count] = row[0]
                row[0] = emotion_count
                row = np.array(row)
                row = row.astype(np.float64)
                #for n in range(len(row)):
                #    if row[n] == 0:
                #        find_zeros[n] += 1
                data_np[emotion_count] = row
                data_x[emotion_count] = row[2:len(row)] #comment out to save RAM
                data_y.append(row[0]) #comment out to save RAM
                non_selected_emotion_indexes.append(emotion_count)
                emotion_count += 1
                non_selected_emotion_count += 1
                #data.append(row)
            if emotion_count % 1000 == 0:
                print(str(emotion_count), "data points added to data_np")
print(str(emotion_count), "audio files extracted from csv file")
print('audio files of emotion "' + selected_emotion + '" labeled as 1, all other audio files labeled as 0')

#makes new np array balanced_data, which has all the audio files of the selected emotion, and an equal amount of audio files of non-selected emotions
#balanced_data = np.zeros((selected_emotion_count * 2, 6374), dtype = np.float64) #for compare2016.csv
balanced_data = np.zeros((selected_emotion_count * 2, 386), dtype = np.float64) #for MSP_Podcast_data_with_gender.csv
free_indexes = []
for n in range(len(balanced_data)):
    free_indexes.append(n)
free_index = free_indexes.pop(random.randrange(len(free_indexes)))
assert selected_emotion_count < non_selected_emotion_count
for n in selected_emotion_indexes: #puts all audio files of the selected emotion into balanced_data
    free_index = free_indexes.pop(random.randrange(len(free_indexes))) #sets free index to an available index
    balanced_data[free_index] = data_np[n]
non_selected_emotion_indexes_copy = non_selected_emotion_indexes.copy()
while len(free_indexes) > 0: #fills the remaining rows of balanced_data with audio files of non-selected emotions
    free_index = free_indexes.pop(random.randrange(len(free_indexes)))
    n = non_selected_emotion_indexes_copy.pop(random.randrange(len(non_selected_emotion_indexes_copy)))
    balanced_data[free_index] = data_np[n]
non_selected_emotion_indexes_copy = None #trash collection
print("np array balanced_data, which has all audio files of the selected emotion, and an equal amount of non-selected emotions, has been created")

#data_np = None #trash collection - uncomment if low on RAM

#divide data into x (attributes) and y (class/emotion)
np.random.shuffle(balanced_data)
balanced_data_x = balanced_data[:,2:]
balanced_data_x = preprocessing.normalize(balanced_data_x)
scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(balanced_data_x)
balanced_data_x = scaler.transform(balanced_data_x)
balanced_data_y = balanced_data[:,1]
print("data scaled and split into attributes (balanced_data_x) and labeles (balanced_data_y)")

#RAM efficient partitioning
#create train_x, train_y, test_x, and test_y np arrays
data_partition = int(len(balanced_data) * 0.8)
attributes_len = len(balanced_data[0]) - 1
train_x = balanced_data_x[0:data_partition]
train_y = balanced_data_y[0:data_partition]
test_x = balanced_data_x[data_partition:len(balanced_data)]
test_y = balanced_data_y[data_partition:len(balanced_data)]
print("np arrays train_x, train_y, test_x, and test_y created out of balanced_data")


#training logistic regression
model = LogisticRegression(random_state = 0)

#training svm
#model = svm.SVC(kernel = "linear", cache_size = 7000)
#model = svm.LinearSVC(C = 1.0, max_iter = 1_000)
#model = svm.SVC(kernel = "rbf", gamma = 0.7, C = 1.0, cache_size = 7000)
#model = svm.SVC(kernel = "poly", degree = 3, C = 1.0, cache_size = 7000)

#training random forest
#n_estim = 256
#crit = "entropy"
#max_feat = 500
#max_dep = 8
#model = RandomForestClassifier(n_estimators = n_estim,
#                            criterion = crit,
#                            max_features = max_feat,
#                            max_depth = max_dep)
print("model made")

model.fit(train_x, train_y)
print("model fitted")

#saving model
filename = "binary_classifier_for_" + selected_emotion + ".joblib"
joblib.dump(model, filename)
print('model saved as "' + filename + '"')

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
    prediction_test = model.predict(x)
    for n in range(25):
        print("real value:", str(y[n]), ", predicted value:", str(prediction_test[n]))
    accuracy = accuracy_score(y_true = y, y_pred = prediction_test)
    precision = precision_score(prediction_test, y)
    recall = recall_score(prediction_test, y)
    count_neg_and_pos(prediction_test)
    print("Accuracy:", str(accuracy))
    print("Precision:", str(precision))
    print("Recall:", str(recall))

test(test_x, test_y)

#trash collection to free up RAM
train_x = None
train_y = None
test_x = None
test_y = None
print("train_x, train_y, test_x, and test_y set to None to free up RAM")

#to test whole data set
"""
data_y = data_np[:,0]
data_np = data_np[:,1:]
data_np = preprocessing.normalize(data_np)
scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(data_np)
data_np = scaler.transform(data_np)
"""
np.random.shuffle(data_np)
test(data_np[:,2:], data_np[:,1])

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#sklearn feature importance and ranks for random forest -> model.feauture_importances_
#sklearn feature importance for svm -> model.coef_

