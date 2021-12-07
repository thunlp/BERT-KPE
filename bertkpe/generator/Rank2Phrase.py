import logging
import numpy as np
from .generator_utils import (
    remove_empty,
    remove_empty_phase,
    del_stemming_duplicate_phrase,
)

logger = logging.getLogger()


def rank2phrase(examples, logit_lists, indices, stem_flag=False, return_num=None):
    batch_predictions = []
    for batch_id, logit_list in enumerate(logit_lists):
        example = examples[indices[batch_id]]

        params = {"gram_list": example["phrase_list"], "score_logits": logit_list}

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


def decode_n_best_candidates(gram_list, score_logits):

    assert len(gram_list) == len(score_logits)
    ngrams = [(gram.split(), score) for gram, score in zip(gram_list, score_logits)]
    sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)

    return sorted_ngrams
