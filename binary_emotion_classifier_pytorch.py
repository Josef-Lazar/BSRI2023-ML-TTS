import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import random

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#each index is a audio file name, and its corresponding value is the partition (train, test1, test2, development) it is in
data_partition_dictionary = {}
#partitions_reader = open("/home/josef/Desktop/BSRI/Partitions.txt") #on linux
partitions_reader = open("C:/Users/pepou/OneDrive/Desktop/Folders/Classes/y4s0.5/BSRI/Partitions.txt") #on windows
for line in partitions_reader:
    if len(line) > 1:
        partition_and_name = line.split("; ")
        partition = partition_and_name[0]
        name = partition_and_name[1]
        name = name[0:len(name) - 1] #gets rid of "\n" at the end of each name
        #print(name)
        data_partition_dictionary[name] = partition
print("data partition dictionary created")

#eliminate audiofiles whose emotional label is "X", and copy to podcast_data list
podcast_data = []
selected_emotion = "H"
selected_emotion_count = 0
#with open("/home/josef/Desktop/BSRI/MSP_Podcast_data.csv") as data: #on linux
with open("C:/Users/pepou/OneDrive/Desktop/Folders/Classes/y4s0.5/BSRI/MSP_Podcast_data.csv") as data: #on windows
    data_reader = csv.reader(data)
    for row in data:
        row = row.split(",")
        row = row[0:len(row) - 1]
        #if row[1] != "X" and (row[1] != "N" or random.randint(0, 10) < 1) and (row[1] != "H" or random.randint(0, 5) < 1) and row[0] != "@attribute name string":
        if row[1] != "X" and row[0] != "@attribute name string":
            podcast_data.append(row)
            if row[1] == selected_emotion:
                selected_emotion_count += 1
            #print(row)
print("podcast data copied from csv file to podcast_data list")

#reduce non-selected emotion count to that of selected emotion count
while len(podcast_data) > 2 * selected_emotion_count:
    index = random.randint(0, len(podcast_data) - 1)
    if podcast_data[index][1] != selected_emotion:
        podcast_data.pop(index)
print("size of non-selected emotion samples reduced to size of selected emotion samples")

#sort audio files from podcast_data into appropriate partition list
train = []
test1 = []
test2 = []
development = []
for row in podcast_data:
    if row[1] == selected_emotion:
        row[1] = 1
    else:
        row[1] = 0
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
print("audio files from podcast_data sorted into train, test1, test2, and development lists")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train = np.array(train)
train = train[:,1:]
np.random.shuffle(train)
train = train.astype(float)
train_labels = train[:, 0]
train_labels = torch.tensor(train_labels, device = device)
train_labels = train_labels.to(torch.float32)
train_attributes = train[:, 1:]
train_attributes = torch.tensor(train_attributes, device = device)
train_attributes = train_attributes.to(torch.float32)

test1 = np.array(test1)
test1 = test1[:,1:]
np.random.shuffle(test1)
test1 = test1.astype(float)
test1_labels = test1[:, 0]
test1_labels = torch.tensor(test1_labels, device = device)
test1_labels = test1_labels.to(torch.float32)
test1_attributes = test1[:, 1:]
test1_attributes = torch.tensor(test1_attributes, device = device)
test1_attributes = test1_attributes.to(torch.float32)

test2 = np.array(test2)
test2 = test2[:,1:]
np.random.shuffle(test2)
test2 = test2.astype(float)
test2_labels = test2[:, 0]
test2_labels = torch.tensor(test2_labels, device = device)
test2_attributes = test2[:, 1:]
test2_attributes = torch.tensor(test2_attributes, device = device)

development = np.array(development)
development = development[:,1:]
np.random.shuffle(development)
development = development.astype(float)
development_labels = development[:, 0]
development_labels = torch.tensor(development_labels, device = device)
development_attributes = development[:, 1:]
development_attributes = torch.tensor(development_attributes, device = device)
print("train, test1, test2, and development lists sorted into lable and attribute lists and converted to pytorch tensors")

batch_size = 512
train_dataset = TensorDataset(train_attributes, train_labels)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size)
test_dataset = TensorDataset(test1_attributes, test1_labels)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            #nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    count = 0
    for batch in dataloader:
        X, y = batch
        pred = model(X)
        y = y.long()
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if count % 100 == 0:
            loss, current = loss.item(), count * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            count += 1

def test_model(dataloader, model, loss_fn):
    true_positives = [0] * 2 # true positives for each number
    false_positives = [0] * 2 # false positives for each number
    false_negatives = [0] * 2 # false negatives foe each number
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y = y.long()
            pred = model(X)
            #print(pred.argmax(1))
            #print(y)
            #print("--------------------------")
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in range(len(y)):
                if pred.argmax(1)[i] == y[i]:
                    true_positives[y[i]] += 1
                else:
                    false_positives[pred.argmax(1)[i]] += 1
                    false_negatives[y[i]] += 1
                
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    display_confusion_matrix(true_positives, false_positives, false_negatives)

def display_confusion_matrix(tp, fp, fn):
    assert len(fp) == len(fn)
    precision = [0] * 2
    recall = [0] * 2
    print("number | precision | recall")
    for i in range(len(fp)):
        if tp[i] + fp[i] == 0:
            precision[i] = 0
        else:
            precision[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall[i] = 0
        else:
            recall[i] = tp[i] / (tp[i] + fn[i])
        print(str(i) + " | " +  str(precision[i]) + " | " + str(recall[i]))
    average_precision = sum(precision) / len(precision)
    average_recall = sum(recall) / len(recall)
    print("average | " + str(average_precision) + " | " + str(average_recall))
    print()

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_model(train_dataloader, model, loss_fn, optimizer)
    test_model(test_dataloader, model, loss_fn)
print("Done!")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


