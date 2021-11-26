"""Split the json data in train, test and validation data"""

from sklearn.model_selection import train_test_split
import os

# read the json file
with open("data.json") as f:
    lines = f.readlines()

#  split the data into train, test and validation
train, test = train_test_split(lines, test_size=0.3)
val, test = train_test_split(test, test_size=0.5)

# path to target folder
data_path = "../data/prepro_dataset/multidata/"
os.makedirs(data_path, exist_ok=True)

# write to train, test and validation files
with open(data_path + "wikinews.train.json", "w") as f:
    for line in train:
        f.write(line)

with open(data_path + "wikinews.test.json", "w") as f:
    for line in test:
        f.write(line)

with open(data_path + "wikinews.dev.json", "w") as f:
    for line in val:
        f.write(line)
