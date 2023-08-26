import csv
import pandas

#make list of out of "name_order.txt
#name_order_reader = open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/train-clean-360_audio_file_paths.txt", "r")
name_order_reader = open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/test-clean_audio_file_paths.txt", "r")
name_order_list = []
for line in name_order_reader:
    #name = line.split("/")[-1]
    name = line.replace("\n", "")
    name_order_list.append(name)

#read chroma.csv and make column list and data list
columns = []
audio_file_data = []
with open("/home/josef/Desktop/BSRI/opensmile-3.0-linux-x64/chroma.csv", newline = "") as chroma:
    chroma_reader = csv.reader(chroma)
    for row in chroma_reader:
        if len(row) == 1 and row[0] != "@data" and row[0] != "@relation openSMILE_features":
            columns.append(row[0])
        elif len(row) == 386:
            audio_file_data.append(row)
    #columns.insert(1, "emotion label")
print('data from "chroma.csv" has been read into columns list and audio_file_data list')

#make combined csv with correct column lables, and audiofile names next to attributes
count = 0
with open("libritts_data.csv", "w", newline = "") as libritts_data:
    libritts_data_writer = csv.writer(libritts_data)
    libritts_data_writer.writerow(columns)
    for n in range(len(audio_file_data)):
        row = audio_file_data[n]
        row[0] = name_order_list[n]
        libritts_data_writer.writerow(row)
        count += 1
        if count % 1000 == 0:
            print(str(count), "out of", str(len(audio_file_data)), "done")
print('all data from "chroma.csv" has been reformated and written to "libritts_data.csv"')


