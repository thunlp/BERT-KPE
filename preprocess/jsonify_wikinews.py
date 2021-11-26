import json
import os
import pdb
import string
import spacy
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("spanish")
nlp = spacy.load("es_core_news_sm")

passage_folder = "../data/dataset/wicc/docsutf8/"
keyphrase_folder = "../data/dataset/wicc/keys/"
dest_file = "./data.json"
file_list = os.listdir(passage_folder)

num = 0
miss_num = 0


def lemmatizer(text):
    doc = nlp(text[0])
    return " ".join([word.lemma_ for word in doc])


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_fullstop(text):
    return text.replace(".", "")


def find_sequence(seq, _list):
    seq_list = seq
    all_occurrence = [
        idx
        for idx in [i for i, x in enumerate(_list) if x == seq_list[0]]
        if seq_list == _list[idx : idx + len(seq_list)]
    ]
    return -1 if not all_occurrence else all_occurrence


def lower_list(_list):
    return [stemmer.stem(x.lower()) for x in _list]


with open(dest_file, "a") as jsonfile:
    for file_name in file_list:

        print(file_name)
        a_dict = {}
        with open(passage_folder + file_name, "r") as f:
            text = f.read()
            text = remove_fullstop(text)
            # text = lemmatizer(text)
            str_en = text.encode("ascii", "ignore")
            str_de = str_en.decode()
            a_dict["doc_words"] = str_de.split()
        base = os.path.splitext(file_name)[0]
        with open(keyphrase_folder + base + ".key", "r") as f:
            text = f.readlines()
            keyphrase_list = []

            for line in text:
                line = line.strip("\n")
                if line:
                    str_en = line.encode("ascii", "ignore")
                    str_de = str_en.decode()
                    keyphrase_list.append(str_de.split())

        num += 1
        pos_list = []
        for keyphrase in keyphrase_list:
            kp_pos_list = []
            ind_list = find_sequence(
                lower_list(keyphrase), lower_list(a_dict["doc_words"])
            )
            if ind_list == -1:
                miss_num += 1
                keyphrase_list.remove(keyphrase)
                continue
                # pdb.set_trace()
            else:
                # print(keyphrase)
                # print(a_dict["doc_words"])
                for ind in ind_list:
                    start_pos = ind
                    end_pos = ind + len(keyphrase) - 1
                    kp_pos_list.append([start_pos, end_pos])
                pos_list.append(kp_pos_list)

        if keyphrase_list != []:
            a_dict["keyphrases"] = keyphrase_list
            a_dict["start_end_pos"] = pos_list
            a_dict["url"] = file_name  # This acts as identifier while evaluating
            jsonString = json.dump(a_dict, jsonfile)
            jsonfile.write("\n")

# print(num, miss_num)
