import logging
import numpy as np
from ..constant import Tag2Idx
from .generator_utils import remove_empty_phase, del_stemming_duplicate_phrase

logger = logging.getLogger()


def chunk2phrase(
    examples, logit_lists, indices, max_phrase_words, return_num, stem_flag=False
):
    batch_predictions = []
    for batch_id, logit_list in enumerate(logit_lists):
        example = examples[indices[batch_id]]

        params = {
            "orig_tokens": example["doc_words"],
            "gram_logits": logit_list,
            "max_gram": max_phrase_words,
        }

        n_best_phrases_scores = decode_n_best_candidates(**params)
        candidate_KP, score_KP = remove_empty_phase(n_best_phrases_scores)

        if return_num:
            if stem_flag:
                candidate_KP, score_KP = del_stemming_duplicate_phrase(
                    candidate_KP, score_KP, return_num
                )
            else:
                candidate_KP = candidate_KP[:return_num]
                score_KP = score_KP[:return_num]
            assert len(candidate_KP) == return_num

        assert len(candidate_KP) == len(score_KP)
        batch_predictions.append((example["url"], candidate_KP, score_KP))
    return batch_predictions


def decode_n_best_candidates(orig_tokens, gram_logits, max_gram):
    """
    max_gram :  type :int , max_phrase_words
    return : phrase token list & score list
    """
    orig_tokens = [token.lower() for token in orig_tokens]
    sorted_ngrams = decode_ngram(
        orig_tokens=orig_tokens, gram_logits=gram_logits, max_gram=max_gram
    )
    return sorted_ngrams


def decode_ngram(orig_tokens, gram_logits, max_gram):

    ngram_score = []
    for n in range(max_gram):
        for i in range(len(orig_tokens) - n):
            ngram_score.append(
                (" ".join(orig_tokens[i : i + n + 1]), gram_logits[n][i])
            )

    phrase_set = {}
    for n_gram, n_gram_score in ngram_score:
        if n_gram not in phrase_set or n_gram_score > phrase_set[n_gram]:
            phrase_set[n_gram] = n_gram_score
        else:
            continue

    phrase_list = []
    for phrase, score in phrase_set.items():
        phrase_list.append((phrase.split(), score))

    sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
    return sorted_phrase_list
