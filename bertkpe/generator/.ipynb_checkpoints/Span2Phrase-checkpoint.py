import logging
import numpy as np
import torch.nn.functional as F
from .generator_utils import remove_empty_phase, del_stemming_duplicate_phrase

logger = logging.getLogger()


def span2phrase(
    examples,
    start_lists,
    end_lists,
    indices,
    max_phrase_words,
    return_num=None,
    stem_flag=False,
):

    batch_predictions = []
    for batch_id, (start_logit, end_logit) in enumerate(zip(start_lists, end_lists)):
        example = examples[indices[batch_id]]

        assert (
            len(start_logit)
            == len(end_logit)
            == len(end_logit[0])
            == len(example["doc_words"])
        )  # word_len

        params = {
            "orig_tokens": example["doc_words"],
            "start_logit": start_logit,
            "end_logit": end_logit,
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


def decode_n_best_candidates(orig_tokens, start_logit, end_logit, max_gram):
    """
    max_gram :  type :int , max_phrase_words
    return : phrase token list & score list
    """
    assert len(orig_tokens) == len(start_logit) == len(end_logit)
    orig_tokens = [token.lower() for token in orig_tokens]

    sorted_ngrams = decode_span2phrase(
        **{
            "orig_tokens": orig_tokens,
            "start_logit": start_logit,
            "end_logit": end_logit,
            "max_gram": max_gram,
        }
    )
    return sorted_ngrams


def decode_span2phrase(orig_tokens, start_logit, end_logit, max_gram):
    phrase2score = {}
    for (i, s) in enumerate(start_logit):
        for (j, e) in enumerate(end_logit[i][i : (i + max_gram)]):
            phrase = " ".join(orig_tokens[i : (i + j + 1)])
            score = s * e
            if (phrase not in phrase2score) or (score > phrase2score[phrase]):
                phrase2score[phrase] = score
            else:
                continue

    phrase_list = []
    for phrase, score in phrase2score.items():
        phrase_list.append((phrase.split(), score))

    sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
    return sorted_phrase_list
