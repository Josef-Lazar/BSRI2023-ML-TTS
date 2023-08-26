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

#copies MSP_Podcast_data_liwc_and_features.csv to data and shuffles it
data_reader = open("/home/josef/Desktop/BSRI/MSP_Podcast_data_liwc_and_features.csv", "r") #ubuntu
#data_reader = open("C:/Users/pepou/OneDrive/Desktop/Folders/Classes/y4s0.5/BSRI/MSP_Podcast_data_liwc_and_features.csv", "r") #windows
data = list(csv.reader(data_reader, delimiter = ","))
data_reader.close()
data.pop(0) #remove column labels
random.shuffle(data)

def eliminate_X():
    n = len(data) - 1
    while n >= 0:
        if data[n][2] == "X":
            data.pop(n)
        n -= 1

def eliminate_non_female():
    n = len(data) - 1
    while n >= 0:
        if data[n][3] != "Female":
            data.pop(n)
        n -= 1

def balance_data(data):
    selected_emotion_count = 0
    selected_emotion_indexes = []
    non_selected_emotion_count = 0
    non_selected_emotion_indexes = []
    for n in range(len(data)):
        if data[n][2] == selected_emotion:
            selected_emotion_count += 1
            selected_emotion_indexes.append(n)
        else:
            non_selected_emotion_count += 1
            non_selected_emotion_indexes.append(n)
    assert selected_emotion_count <= non_selected_emotion_count
    balanced_data = [0] * selected_emotion_count * 2
    available_indexes = []
    for n in range(len(balanced_data)):
        available_indexes.append(n)
    for n in selected_emotion_indexes:
        balanced_data_index = available_indexes.pop(random.randrange(len(available_indexes)))
        balanced_data[balanced_data_index] = data[n]
    for n in range(len(balanced_data)):
        if balanced_data[n] == 0:
            data_index = non_selected_emotion_indexes.pop(random.randrange(len(non_selected_emotion_indexes)))
            balanced_data[n] = data[data_index]
    return balanced_data

def split_and_scale(data):
    x = []
    y = []
    audiofile_names = []
    partition = []
    for n in range(len(data)):
        if data[n][2] == selected_emotion:
            y.append(1.0)
        else:
            y.append(0.0)
        x.append(data[n][5:389] + data[n][390:len(data[0])]) #data[n][389] is left out because it is corrupted
        audiofile_names.append(data[n][0])
        partition.append(data[n][4])
    #scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(x)
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    return x, y, audiofile_names, partition

def partition(x, y, partition):
    train_x = []
    train_y = []
    test1_x = []
    test1_y = []
    test2_x = []
    test2_y = []
    development_x = []
    development_y = []
    assert len(x) == len(y)
    assert len(x) == len(partition)
    n = len(x) - 1
    while n >= 0:
        if partition[n] == "Train":
            train_x.append(x[n])
            train_y.append(y[n])
        elif partition[n] == "Test1":
            test1_x.append(x[n])
            test1_y.append(y[n])
        elif partition[n] == "Test2":
            test2_x.append(x[n])
            test2_y.append(y[n])
        elif partition[n] == "Development":
            development_x.append(x[n])
            development_y.append(y[n])
        else:
            print("wtf")
            assert False
        n -= 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test1_x = np.array(test1_x)
    test1_y = np.array(test1_y)
    test2_x = np.array(test2_x)
    test2_x = np.array(test2_y)
    development_x = np.array(development_x)
    development_y = np.array(development_y)
    return train_x, train_y, test1_x, test1_y, test2_x, test2_y, development_x, development_y

def shrink_data(data, percentage):
    remove_count = len(data) * percentage
    while remove_count > 0:
        data.pop(random.randrange(len(data)))
        remove_count -= 1
    return data

print(len(data))
eliminate_non_female()
print(len(data))
#data = shrink_data(data, 0.99)
#print(len(data))
selected_emotion = "H"
data_x, data_y, audiofile_names, data_partition = split_and_scale(data)

model_file_name = "binary_svm_poly_deg3_for_H.joblib"
model = joblib.load(model_file_name)
print("model loaded")

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
    with open("/home/josef/BSRI/H_files.txt", "w") as writer:
        for n in range(len(prediction_test)):
            if prediction_test[n] == 1.0:
                audiofile_name = audiofile_names[n]
                writer.write("%s\n" % audiofile_name)
        print(selected_emotion, "files saved to H_files.txt")
    for n in range(25):
        print("real value:", str(y[n]), ", predicted value:", str(prediction_test[n]))
    accuracy = accuracy_score(y_true = y, y_pred = prediction_test)
    precision = precision_score(prediction_test, y)
    recall = recall_score(prediction_test, y)
    count_neg_and_pos(prediction_test)
    print("Accuracy:", str(accuracy))
    print("Precision:", str(precision))
    print("Recall:", str(recall))

test(data_x, data_y)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)








