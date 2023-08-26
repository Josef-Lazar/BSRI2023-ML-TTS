import csv
import pandas

features_reader = open("/home/josef/Desktop/BSRI/libritts_data.csv", "r")
features_data = list(csv.reader(features_reader, delimiter = ","))
features_reader.close()
print("libritts_data.csv copied to features_data list")

liwc_reader = open("/home/josef/Desktop/BSRI/liwc.csv", "r")
liwc_data = list(csv.reader(liwc_reader, delimiter = ","))
liwc_reader.close()
print("liwc_features_libritts.csv copied to liwc_data list")

#make dictionaries that take libritts audiofile ids return sex and name
id_to_sex = {}
id_to_name = {}
#libritts_metadata_reader = open("/home/josef/Extreme_SSD/train-clean-360/LibriTTS/SPEAKERS.txt", "r")
#libritts_metadata_reader = open("/home/josef/Extreme_SSD/test-clean/LibriTTS/SPEAKERS.txt", "r")
libritts_metadata_reader = open("/home/josef/Extreme_SSD/dev-clean/LibriTTS/SPEAKERS.txt", "r")
for line in libritts_metadata_reader:
    if line[0] != ";":
        line_split = line.split("|")
        id = int(line_split[0])
        sex = line_split[1][1]
        assert sex == "M" or sex == "F"
        name = line_split[-1]
        name_len = len(name)
        name = name[1:name_len] #gets rid of space at the start
        name = name.split("\n")[0]
        id_to_sex[id] = sex
        id_to_name[id] = name
libritts_metadata_reader.close()

#make dictionary that takes libritts audiofile path and returns corresponding row in liwc_data
audiofile_name_to_row = {}
for n in range(len(liwc_data)):
    audiofile_name = liwc_data[n][0]
    audiofile_name_to_row[audiofile_name] = n

#combine features and liwc and metadata
assert len(features_data) <= len(liwc_data)
#path, text, sex, id, name, features, liwc
columns = ["path", "text", "sex", "id", "name"] + features_data[0][1:len(features_data[0])] + liwc_data[0][2:len(liwc_data[0])]
combined_data = [columns]
for n in range(1, len(features_data)):
    row = features_data[n]
    audiofile_name = row[0].split("/")[-1]
    audiofile_name = audiofile_name.split(".wav")[0]
    liwc_row_index = audiofile_name_to_row[audiofile_name]
    liwc_row = liwc_data[liwc_row_index]
    text = liwc_row[1]
    id = int(audiofile_name.split("_")[0])
    sex = id_to_sex[id]
    name = id_to_name[id]
    row.insert(1, text)
    row.insert(2, sex)
    row.insert(3, id)
    row.insert(4, name)
    row = row + liwc_row[2:len(liwc_row)]
    combined_data.append(row)
    if n % 1000 == 0:
        print(str(n + 1), "out of", str(len(features_data)), "completed")
print(str(n + 1), "out of", str(len(features_data)), "completed")

#save combined_data as "LibriTTS_liwc_and_features.csv"
with open("LibriTTS_liwc_and_features.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerows(combined_data)
print('combined data seved as "LibriTTS_liwc_and_features.csv"')









