import re
import string
import unicodedata

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


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


def remove_empty_phase(phrases_scores):
    phrase_list = []
    score_list = []
    for (
        phrase,
        score,
    ) in phrases_scores:
        if len(phrase) > 0:
            if len(phrase[0]) > 0:
                phrase_list.append(normalize_answer(phrase))
                score_list.append(score)
    return phrase_list, score_list


def stem_norm_phrase(phrase):
    norm_chars = unicodedata.normalize("NFD", phrase)
    stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
    return norm_chars, stem_chars


def del_stemming_duplicate_phrase(phrase_list, score_list, return_num):
    tot_phrases_set = set()
    return_phrases, return_scores = [], []

    for phrase, score in zip(phrase_list, score_list):
        norm_phrase, stem_phrase = stem_norm_phrase(phrase)

        if (norm_phrase not in tot_phrases_set) and (
            stem_phrase not in tot_phrases_set
        ):
            return_phrases.append(phrase)
            return_scores.append(score)

        tot_phrases_set.add(norm_phrase)
        tot_phrases_set.add(stem_phrase)

        if len(return_phrases) >= return_num:
            break
    return return_phrases, return_scores
