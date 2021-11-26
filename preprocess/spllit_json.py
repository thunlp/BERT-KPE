"""Split the json data in train, test and validation data"""

from sklearn.model_selection import train_test_split
import json
import os

with open("data.json") as f:
    lines = f.readlines()

train, test = train_test_split(lines, test_size=0.3)
val, test = train_test_split(test, test_size=0.5)

data_path = "../data/prepro_dataset/wikinews/"
os.makedirs(data_path, exist_ok=True)

with open(data_path + "wikinews.train.json", "w") as f:
    for line in train:
        f.write(line)

with open(data_path + "wikinews.test.json", "w") as f:
    for line in test:
        f.write(line)

with open(data_path + "wikinews.dev.json", "w") as f:
    for line in val:
        f.write(line)
