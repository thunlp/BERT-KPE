import sys
import json
import re
import string
import logging
import numpy as np
import unicodedata

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

logger = logging.getLogger()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return " ".join([lower(x) for x in s]).rstrip()


def remove_empty(a_list):
    new_list = []
    for i in a_list:
        if len(i) > 0:
            if len(i[0]) > 0:
                new_list.append(normalize_answer(i))
    return new_list


# ----------------------------------------------------------------------------
# stem phrase
def norm_and_stem(phrase_list, merge=False):

    norm_stem_phrases = []
    for phrase in phrase_list:
        norm_chars = unicodedata.normalize("NFD", phrase)
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
        if merge:
            norm_stem_phrases.append(norm_chars)
            norm_stem_phrases.append(stem_chars)
        else:
            norm_stem_phrases.append((norm_chars, stem_chars))
    return norm_stem_phrases


def get_match_scores(pred_list, truth_list):
    match_score = np.asarray([0.0] * len(pred_list), dtype="float32")

    norm_stem_preds = norm_and_stem(pred_list)
    norm_stem_truths = norm_and_stem(truth_list, merge=True)

    for pred_id, pred_seq in enumerate(norm_stem_preds):
        if pred_seq[0] in norm_stem_truths or pred_seq[1] in norm_stem_truths:
            match_score[pred_id] = 1
    return match_score


# ----------------------------------------------------------------------------


def evaluate(candidates, references, urls):
    precision_scores, recall_scores, f1_scores = (
        {1: [], 3: [], 5: [], 10: []},
        {1: [], 3: [], 5: [], 10: []},
        {1: [], 3: [], 5: [], 10: []},
    )
    for url in urls:
        candidate = remove_empty(
            candidates[url]["KeyPhrases"]
        )  # covert word list to string
        reference = remove_empty(references[url]["KeyPhrases"])  # have remove empty

        # stem match scores
        match_list = get_match_scores(candidate, reference)

        # Padding
        if len(match_list) < 10:
            for _ in range(10 - len(match_list)):
                candidate.append("")
            assert len(candidate) == 10

        for topk in [1, 3, 5, 10]:

            # Micro-Averaged  Method
            micropk = (
                float(sum(match_list[:topk])) / float(len(candidate[:topk]))
                if len(candidate[:topk]) > 0
                else 0.0
            )
            micrork = (
                float(sum(match_list[:topk])) / float(len(reference))
                if len(reference) > 0
                else 0.0
            )

            if micropk + micrork > 0:
                microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
            else:
                microf1 = 0.0

            precision_scores[topk].append(micropk)
            recall_scores[topk].append(micrork)
            f1_scores[topk].append(microf1)

    return f1_scores, precision_scores, recall_scores


def files_are_good(candidate, reference):
    referenceURLs = set(reference.keys())
    candidateURLs = set(candidate.keys())
    if len((referenceURLs - candidateURLs)) > 0:
        logger.info(
            "ERROR:Candidate File is missing URLS present in reference file\nMissing urls:{}".format(
                referenceURLs - candidateURLs
            )
        )
        return False
    if len((candidateURLs - referenceURLs)) > 0:
        logger.info(
            "ERROR:Candidate File includes URLS not present in reference file\nUnexpected urls:{}".format(
                candidateURLs - referenceURLs
            )
        )
        return False
    return True


def load_file(filename):
    data = {}
    with open(filename, "r") as f:
        for l in f:
            item = json.loads(l)
            data[item["url"]] = item
    return data


def evaluate_kp20k(candidate, reference_filename):
    reference = load_file(reference_filename)
    if files_are_good(candidate, reference) == True:
        candidate_urls = set(candidate.keys())
        reference_urls = set(reference.keys())
        urls = reference_urls.intersection(candidate_urls)
        f1_scores, precision_scores, recall_scores = evaluate(
            candidate, reference, urls
        )
        return f1_scores, precision_scores, recall_scores
    else:
        logger.info(
            "Candidate file and Reference are not comparable. Please verify your candidate file."
        )
