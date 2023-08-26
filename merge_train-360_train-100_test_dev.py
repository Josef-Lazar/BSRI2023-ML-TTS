import csv
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

train_clean_360_reader = open("/home/josef/Desktop/BSRI/LibriTTS_liwc_and_features_train-clean-360.csv", "r")
train_clean_360_data = list(csv.reader(train_clean_360_reader, delimiter = ","))
train_clean_360_reader.close()
print("train_clean_360 loaded in")

train_clean_100_reader = open("/home/josef/Desktop/BSRI/LibriTTS_liwc_and_features_train-clean-100.csv", "r")
train_clean_100_data = list(csv.reader(train_clean_100_reader, delimiter = ","))
train_clean_100_reader.close()
print("train_clean_100 loaded in")

test_clean_reader = open("/home/josef/Desktop/BSRI/LibriTTS_liwc_and_features_test-clean.csv", "r")
test_clean_data = list(csv.reader(test_clean_reader, delimiter = ","))
test_clean_reader.close()
print("test_clean loaded in")

dev_clean_reader = open("/home/josef/Desktop/BSRI/LibriTTS_liwc_and_features_dev-clean.csv", "r")
dev_clean_data = list(csv.reader(dev_clean_reader, delimiter = ","))
dev_clean_reader.close()
print("dev_clean loaded in")

columns = train_clean_360_data[0]
columns.insert(1, "data set")
with open("LibriTTS_liwc_and_features_all.csv", "w", newline = "") as libritts_data:
    libritts_data_writer = csv.writer(libritts_data)
    libritts_data_writer.writerow(columns)
print("column labels written to LibriTTS_liwc_and_features_all.csv")

all_len = len(train_clean_360_data) + len(train_clean_100_data) + len(test_clean_data) + len(dev_clean_data) - 4
for n in range(1, len(train_clean_360_data)):
    row = train_clean_360_data[n]
    row.insert(1, "train-clean-360")
    with open("LibriTTS_liwc_and_features_all.csv", "a", newline = "") as libritts_data:
        podcast_data_writer = csv.writer(libritts_data)
        podcast_data_writer.writerow(row)
    if n % 1000 == 0:
        print(str(n), "out of", str(all_len), "done")
print("train-clean-360 copied")
print("")

current_len = len(train_clean_360_data) - 1
for n in range(1, len(train_clean_100_data)):
    row = train_clean_100_data[n]
    row.insert(1, "train-clean-100")
    with open("LibriTTS_liwc_and_features_all.csv", "a", newline = "") as libritts_data:
        podcast_data_writer = csv.writer(libritts_data)
        podcast_data_writer.writerow(row)
    if (current_len + n) % 1000 == 0:
        print(str(current_len + n), "out of", str(all_len), "done")
print("train-clean-100 copied")
print("")

current_len += len(train_clean_100_data) - 1
for n in range(1, len(test_clean_data)):
    row = test_clean_data[n]
    row.insert(1, "test-clean")
    with open("LibriTTS_liwc_and_features_all.csv", "a", newline = "") as libritts_data:
        podcast_data_writer = csv.writer(libritts_data)
        podcast_data_writer.writerow(row)
    if (current_len + n) % 1000 == 0:
        print(str(current_len + n), "out of", str(all_len), "done")
print("test-clean copied")
print("")

current_len += len(test_clean_data) - 1
for n in range(1, len(dev_clean_data)):
    row = dev_clean_data[n]
    row.insert(1, "dev-clean")
    with open("LibriTTS_liwc_and_features_all.csv", "a", newline = "") as libritts_data:
        podcast_data_writer = csv.writer(libritts_data)
        podcast_data_writer.writerow(row)
    if (current_len + n) % 1000 == 0:
        print(str(current_len + n), "out of", str(all_len), "done")
print("dev-clean copied")
print("")
print("done:)")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
