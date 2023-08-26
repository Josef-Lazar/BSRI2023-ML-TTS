import csv
import pandas
import numpy as np
import os

#make list of out of "name_order.txt
name_order_reader = open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/name_order.txt", "r")
name_order_list = []
for line in name_order_reader:
    name_order_list.append(line.replace("\n", ""))
print("name_oreder list with the audio file names in the order of the chroma.csv file created")

#find emotional label of every audiofile
audio_file_names = []
emotion_labels = []
with open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/labels_consensus.csv", "r") as lables:
    labels_reader = csv.reader(lables)
    for row in labels_reader:
        audio_file_names.append(row[0])
        emotion_labels.append(row[1])
    #so that column labels are not included in data
    audio_file_names.pop(0)
    emotion_labels.pop(0)
print("names and emotions of audio files copied to audio_file_names list and emotion_labels list")

#given an audio file name, the function returns the emotional lable of that file
def find_emotion(audio_file_name):
    index = audio_file_names.index(audio_file_name)
    emotion_lable = emotion_labels[index]
    return emotion_lable

#copies chroma.csv to MSP_Podcast_data_compare2016.csv in correct format
columns = []
audio_file_data = []
iterations = 0
rows = 0
with open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/chroma.csv", newline = "") as chroma:
    chroma_reader = csv.reader(chroma)
    for row in chroma_reader:
        if len(row) == 1 and row[0] != "@data" and row[0] != "@relation openSMILE_features":
            columns.append(row[0])
            iterations += 1
        elif iterations == 6375 and len(row) == 0:
            columns.insert(1, "emotion label")
            with open("MSP_Podcast_data_compare2016.csv", "w", newline = "") as podcast_data:
                podcast_data_writer = csv.writer(podcast_data)
                podcast_data_writer.writerow(columns)
                print("column labels written to MSP_Podcast_data_compate2016.csv")
                rows += 1
            iterations += 1
        elif len(row) == 6375:
            with open("MSP_Podcast_data_compare2016.csv", "a", newline = "") as podcast_data:
                name = name_order_list[rows - 1]
                emotion = find_emotion(name)
                row[0] = name
                row.insert(1, emotion)
                podcast_data_writer = csv.writer(podcast_data)
                podcast_data_writer.writerow(row)
            rows += 1
            if rows % 1000 == 0:
                print(str(rows), "rows from chroma.csv written to MSP_Podcast_data_compare2016.csv")
                total_memory, used_memory, free_memory = map(
                    int, os.popen('free -t -m').readlines()[-1].split()[1:])
                if used_memory/total_memory > 0.9:
                    print("less than 10% of RAM left, kys")
                    assert False
    columns.insert(1, "emotion label")
print("done:)")

