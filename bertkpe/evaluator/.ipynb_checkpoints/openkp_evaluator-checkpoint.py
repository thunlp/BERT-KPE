import sys
import json
import re
import string
import numpy as np
import logging

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


def dedup(kp_list):
    dedupset = set()
    kp_list_dedup = []
    for kp in kp_list:
        if kp in dedupset:
            continue
        kp_list_dedup.append(kp)
        dedupset.add(kp)
    return kp_list_dedup


def get_score_full(candidates, references, maxDepth=5):
    precision = []
    recall = []
    reference_set = set(dedup(references))
    candidates = dedup(candidates)
    referencelen = len(reference_set)
    true_positive = 0
    for i in range(maxDepth):
        if len(candidates) > i:
            kp_pred = candidates[i]
            if kp_pred in reference_set:
                true_positive += 1
        precision.append(true_positive / float(i + 1))
        recall.append(true_positive / float(referencelen))
    return precision, recall


def online_evaluate(candidates, references, urls):
    precision_scores, recall_scores, f1_scores = (
        {1: [], 3: [], 5: []},
        {1: [], 3: [], 5: []},
        {1: [], 3: [], 5: []},
    )
    for url in urls:
        candidate = remove_empty(candidates[url]["KeyPhrases"])
        reference = remove_empty(references[url]["KeyPhrases"])
        p, r = get_score_full(candidate, reference)
        for i in [1, 3, 5]:
            precision = p[i - 1]
            recall = r[i - 1]
            if precision + recall > 0:
                f1_scores[i].append((2 * (precision * recall)) / (precision + recall))
            else:
                f1_scores[i].append(0)
            precision_scores[i].append(precision)
            recall_scores[i].append(recall)
    return f1_scores, precision_scores, recall_scores


def evaluate(candidates, references, urls):
    precision_scores, recall_scores, f1_scores = (
        {1: [], 3: [], 5: []},
        {1: [], 3: [], 5: []},
        {1: [], 3: [], 5: []},
    )
    for url in urls:
        candidate = remove_empty(candidates[url]["KeyPhrases"])
        reference = remove_empty(references[url]["KeyPhrases"]) ## KeyPhrases / Keyphrases
        p, r = get_score_full(candidate, reference)
        for i in [1, 3, 5]:
            precision = p[i - 1]
            recall = r[i - 1]
            if precision + recall > 0:
                f1_scores[i].append((2 * (precision * recall)) / (precision + recall))
            else:
                f1_scores[i].append(0)
            precision_scores[i].append(precision)
            recall_scores[i].append(recall)
    print("########################\nMetrics")
    for i in precision_scores:
        print("@{}".format(i))
        print("F1:{}".format(np.mean(f1_scores[i])))
        print("P:{}".format(np.mean(precision_scores[i])))
        print("R:{}".format(np.mean(recall_scores[i])))
    print("#########################")


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


def evaluate_openkp(candidate, reference_filename):
    reference = load_file(reference_filename)
    if files_are_good(candidate, reference) == True:
        candidate_urls = set(candidate.keys())
        reference_urls = set(reference.keys())
        urls = reference_urls.intersection(candidate_urls)
        f1_scores, precision_scores, recall_scores = online_evaluate(
            candidate, reference, urls
        )
        return f1_scores, precision_scores, recall_scores
    else:
        logger.info(
            "Candidate file and Reference are not comparable. Please verify your candidate file."
        )


def evaluate_multidata(candidate, reference_filename):
    reference = load_file(reference_filename)
    if files_are_good(candidate, reference) == True:
        candidate_urls = set(candidate.keys())
        reference_urls = set(reference.keys())
        urls = reference_urls.intersection(candidate_urls)
        f1_scores, precision_scores, recall_scores = online_evaluate(
            candidate, reference, urls
        )
        return f1_scores, precision_scores, recall_scores
    else:
        logger.info(
            "Candidate file and Reference are not comparable. Please verify your candidate file."
        )


def main(candidate_filename, reference_filename):
    candidate = load_file(candidate_filename)
    reference = load_file(reference_filename)
    if files_are_good(candidate, reference) == True:
        candidate_urls = set(candidate.keys())
        reference_urls = set(reference.keys())
        urls = reference_urls.intersection(candidate_urls)
        evaluate(candidate, reference, urls)
        exit(0)
    else:
        print(
            "Candidate file and Reference are not comparable. Please verify your candidate file."
        )
        exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:evaluate.py <candidate file> <reference file>")
        exit(-1)
    else:
        main(sys.argv[1], sys.argv[2])
