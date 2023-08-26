import csv
import pandas

#make list of out of "name_order.txt
name_order_reader = open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/name_order.txt", "r")
name_order_list = []
for line in name_order_reader:
    name_order_list.append(line.replace("\n", ""))

#find emotional label and gender label of every audiofile
audio_file_names = []
emotion_labels = []
gender_labels = []
with open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/labels_consensus.csv", "r") as lables:
    labels_reader = csv.reader(lables)
    for row in labels_reader:
        audio_file_names.append(row[0])
        emotion_labels.append(row[1])
        gender_labels.append(row[6])
    #so that column labels are not included in data
    audio_file_names.pop(0)
    emotion_labels.pop(0)
    gender_labels.pop(0)

#read chroma.csv and make column list and data list
columns = []
audio_file_data = []
with open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/chroma_IS09_emotion.csv", newline = "") as chroma:
    chroma_reader = csv.reader(chroma)
    for row in chroma_reader:
        if len(row) == 1 and row[0] != "@data" and row[0] != "@relation openSMILE_features":
            columns.append(row[0])
        elif len(row) == 386:
            audio_file_data.append(row)
    columns.insert(1, "emotion label")
    columns.insert(2, "gender label")

#given an audio file name, the function returns the emotional lable of that file
def find_emotion_and_gender(audio_file_name):
    index = audio_file_names.index(audio_file_name)
    emotion_label = emotion_labels[index]
    gender_label = gender_labels[index]
    return emotion_label, gender_label

#make combined csv with correct column lables, and audiofile names next to attributes
count = 0
with open("MSP_Podcast_data_with_gender.csv", "w", newline = "") as podcast_data:
    podcast_data_writer = csv.writer(podcast_data)
    podcast_data_writer.writerow(columns)
    for n in range(len(audio_file_data)):
        row = audio_file_data[n]
        row[0] = name_order_list[n]
        emotion, gender = find_emotion_and_gender(name_order_list[n])
        #emotion = "X"
        row.insert(1, emotion)
        row.insert(2, gender)
        podcast_data_writer.writerow(row)
        count += 1
        if count % 1000 == 0:
            print(str(count), "out of", str(len(audio_file_data)), "done")
print("All done!")


