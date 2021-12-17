"""This script preprocess the multilingual dataset from 
https://github.com/LIAAD/KeywordExtractor-Datasets in desired format
to train a keyphrase extraction model using 
https://github.com/thunlp/BERT-KPE approach"""

import json
import os
from prepro_utils import remove_fullstop, find_sequence, lower_list

# define input and output paths 
# (download data https://github.com/LIAAD/KeywordExtractor-Datasets)
passage_folder = "../data/dataset/wicc/docsutf8/"
keyphrase_folder = "../data/dataset/wicc/keys/"
dest_file = "./multidata.json"
file_list = os.listdir(passage_folder)


with open(dest_file, "a") as jsonfile:
    for file_name in file_list:
        a_dict = {}
        with open(passage_folder + file_name, "r") as f:
            text = f.read()
            # removing fullstops
            text = remove_fullstop(text)
            # converting unicode to ascii
            str_en = text.encode("ascii", "ignore")
            str_de = str_en.decode()
            # adding a list of words to the dictionary
            a_dict["doc_words"] = str_de.split()

        # adding a list of keyphrases to the dictionary
        base = os.path.splitext(file_name)[0]
        with open(keyphrase_folder + base + ".key", "r") as f:
            text = f.readlines()
            keyphrase_list = []

            for line in text:
                line = line.strip("\n")
                if line:
                    # converting unicode to ascii
                    str_en = line.encode("ascii", "ignore")
                    str_de = str_en.decode()
                    keyphrase_list.append(str_de.split())

        # Find occurances of keyphrases in the passage
        pos_list = []
        for keyphrase in keyphrase_list:
            kp_pos_list = []
            ind_list = find_sequence(
                lower_list(keyphrase), lower_list(a_dict["doc_words"])
            )

            # Some keyphrases are not detected in the passage
            if ind_list == -1:
                keyphrase_list.remove(keyphrase)
                continue
            else:
                for ind in ind_list:
                    start_pos = ind
                    end_pos = ind + len(keyphrase) - 1
                    kp_pos_list.append([start_pos, end_pos])
                pos_list.append(kp_pos_list)

        # adding the keyphrases and their positions to the dictionary
        if keyphrase_list != []:
            a_dict["KeyPhrases"] = keyphrase_list
            a_dict["start_end_pos"] = pos_list
            # url acts as identifier while evaluating
            a_dict["url"] = file_name  
            # writing the dictionary to the json file
            jsonString = json.dump(a_dict, jsonfile)
            jsonfile.write("\n")
