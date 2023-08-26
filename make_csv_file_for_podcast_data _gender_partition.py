import csv
import pandas

#makes dictionary that takes audiofile name and returns its partition
partition_dictionary = {}
partition_reader = open("/home/josef/Desktop/BSRI/Partitions.txt", "r")
for line in partition_reader:
    if len(line) > 1:
        partition, name = line.split("; ")
        name = name.split("\n")[0]
        partition_dictionary[name] = partition
partition_dictionary["@attribute name string"] = "partition label"

#deletes everything in MSP_Podcast_data_with_gender_and_partition.csv
clear_csv_file = open("/home/josef/Desktop/BSRI/MSP_Podcast_data_with_gender_and_partition.csv", "w")

#copies MSP_Podcast_data_with_gender.csv to new csv file but with partition labels
count = 0
with open("/home/josef/Desktop/BSRI/MSP_Podcast_data_with_gender.csv", newline = "") as data:
    data_reader = csv.reader(data)
    for row in data_reader:
        name = row[0]
        partition = partition_dictionary[name]
        row.insert(3, partition)
        with open("MSP_Podcast_data_with_gender_and_partition.csv", "a", newline = "") as data_with_partition:
            partition_writer = csv.writer(data_with_partition)
            partition_writer.writerow(row)
            count += 1
            if count % 1000 == 0:
                print(str(count), "done")
print("All done!")


