import logging
import numpy as np
from ..constant import Tag2Idx
from .generator_utils import remove_empty_phase, del_stemming_duplicate_phrase

logger = logging.getLogger()


def tag2phrase(
    examples,
    logit_lists,
    indices,
    max_phrase_words,
    pooling,
    return_num=None,
    stem_flag=False,
):
    batch_predictions = []
    for batch_id, logit_list in enumerate(logit_lists):
        example = examples[indices[batch_id]]

        params = {
            "orig_tokens": example["doc_words"],
            "token_logits": logit_list,
            "max_gram": max_phrase_words,
            "pooling": pooling,
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


def decode_n_best_candidates(orig_tokens, token_logits, max_gram, pooling):
    """
    max_gram :  type :int , max_phrase_words
    return : phrase token list & score list
    """
    assert len(orig_tokens) == len(token_logits)
    orig_tokens = [token.lower() for token in orig_tokens]

    ngrams = []
    for n in range(1, max_gram + 1):
        ngrams.extend(decode_ngram(orig_tokens, token_logits, n, pooling))
    # sorted all n-grams
    sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)
    return sorted_ngrams


def decode_ngram(orig_tokens, token_logits, n, pooling=None):
    """
    Combine n-gram score and sorted
    Inputs :
        n : n_gram
        orig_tokens : document lower cased words' list
        token_logits : each token has five score : for 'O', 'B', 'I', 'E', 'U' tag
        pooling : pooling method :  mean / min / log_mean (min is the best)
        sum_tf : if True Sum All Mention
    Outputs : sorted phrase and socre list
    """
    if n == 1:
        ngram_ids = [Tag2Idx["U"]]
    elif n >= 2:
        ngram_ids = (
            [Tag2Idx["B"]] + [Tag2Idx["I"] for _ in range(n - 2)] + [Tag2Idx["E"]]
        )
    else:
        logger.info("invalid %d-gram !" % n)
    offsets = [i for i in range(len(ngram_ids))]

    # combine n-gram scores
    phrase_set = {}
    valid_length = len(orig_tokens) - n + 1
    for i in range(valid_length):

        n_gram = " ".join(orig_tokens[i : i + n])
        n_gram_score = min(
            [token_logits[i + bias][tag] for bias, tag in zip(offsets, ngram_ids)]
        )

        if n_gram not in phrase_set or n_gram_score > phrase_set[n_gram]:
            phrase_set[n_gram] = n_gram_score
        else:
            continue

    phrase_list = []
    for phrase, score in phrase_set.items():
        phrase_list.append((phrase.split(), score))

    sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
    return sorted_phrase_list
