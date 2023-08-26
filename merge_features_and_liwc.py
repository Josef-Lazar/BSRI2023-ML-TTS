import csv
import pandas

#copies liwc_features_msp-podcast.csv to liwc_data list
liwc_reader = open("/home/josef/Desktop/BSRI/liwc_features_msp-podcast.csv", "r")
liwc_data = list(csv.reader(liwc_reader, delimiter = ","))
liwc_reader.close()
print("liwc_features_msp-podcast.csv copied to liwc_data list")

#makes dictionary that takes audiofile name and returns its index, runs tests, and adds ".wav" to audio file names
name_to_row_liwc_dictionary = {}
for n in range(len(liwc_data)):
    assert len(liwc_data[n]) == 75
    if n > 0:
        assert len(liwc_data[n]) == 75
        if len(liwc_data[n][0]) != 21 and len(liwc_data[n][0]) != 26:
            print(n)
            print(liwc_data[n][0])
        assert len(liwc_data[n][0]) == 21 or len(liwc_data[n][0]) == 26
        liwc_data[n][0] = liwc_data[n][0] + ".wav"
        name_to_row_liwc_dictionary[liwc_data[n][0]] = n
name_to_row_liwc_dictionary["@attribute name string"] = 0
print("name_to_row_liwc_dictionary created")

#copies MSP_Podcast_data_with_gender_and_partition.csv to features_data list
features_reader = open("MSP_Podcast_data_with_gender_and_partition.csv", "r")
features_data = list(csv.reader(features_reader, delimiter = ","))
features_reader.close()
print("MSP_Podcast_data_with_gender_and_partition.csv copied to features_data list")

#combine data
assert len(liwc_data) == len(features_data)
for n in range(len(features_data)):
    assert len(features_data[n]) == 389
    name = features_data[n][0]
    liwc_row_index = name_to_row_liwc_dictionary[name]
    liwc_row = liwc_data[liwc_row_index]
    features_data[n].insert(1, liwc_row[1])
    features_data[n] = features_data[n] + liwc_row[2:len(liwc_row)]
    assert len(features_data[n]) == 463
    liwc_data[liwc_row_index] = None #optional trash collection
    if n % 1000 == 0:
        print(str(n + 1), "out of", str(len(features_data)), "completed")
print(str(n + 1), "out of", str(len(features_data)), "completed")

#save combined data as "MSP_Podcast_data_liwc_and_features.csv"
with open("MSP_Podcast_data_liwc_and_features.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerows(features_data)
print('combined data saved as "MSP_Podcast_data_liwc_and_features.csv"')


