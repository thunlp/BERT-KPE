import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from . import loader_utils
from ..constant import BOS_WORD, EOS_WORD

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# preprocess label
# ------------------------------------------------------------------------------------------
def get_ngram_label(valid_length, start_end_pos, max_phrase_words):
    # flatten, rank, filter overlap for answer positions

    sorted_positions = loader_utils.flat_rank_pos(start_end_pos)
    filter_positions = loader_utils.limit_phrase_length(
        sorted_positions, max_phrase_words
    )

    if len(filter_positions) != len(sorted_positions):
        overlen_flag = True
    else:
        overlen_flag = False

    s_label, e_label = [], []
    for s, e in filter_positions:
        if e < valid_length:
            s_label.append(s)
            e_label.append(e)
        else:
            break
    assert len(s_label) == len(e_label)
    return {"s_label": s_label, "e_label": e_label, "overlen_flag": overlen_flag}


def bert2chunk_preprocessor(
    examples,
    tokenizer,
    max_token,
    pretrain_model,
    mode,
    max_phrase_words,
    stem_flag=False,
):
    logger.info(
        "start preparing (%s) features for bert2chunk (%s) ..." % (mode, pretrain_model)
    )
    overlen_num = 0
    new_examples = []
    for idx, ex in enumerate(tqdm(examples)):

        # tokenize
        tokenize_output = loader_utils.tokenize_for_bert(
            doc_words=ex["doc_words"], tokenizer=tokenizer
        )

        if len(tokenize_output["tokens"]) < max_token:
            max_word = max_token
        else:
            max_word = tokenize_output["tok_to_orig_index"][max_token - 1] + 1

        new_ex = {}
        new_ex["url"] = ex["url"]
        new_ex["tokens"] = tokenize_output["tokens"][:max_token]
        new_ex["valid_mask"] = tokenize_output["valid_mask"][:max_token]
        new_ex["doc_words"] = ex["doc_words"][:max_word]
        assert len(new_ex["tokens"]) == len(new_ex["valid_mask"])
        assert sum(new_ex["valid_mask"]) == len(new_ex["doc_words"])

        if mode == "train":
            parameter = {
                "valid_length": len(new_ex["doc_words"]),
                "start_end_pos": ex["start_end_pos"],
                "max_phrase_words": max_phrase_words,
            }
            # ------------------------------------------------
            label_dict = get_ngram_label(**parameter)
            if label_dict["overlen_flag"]:
                overlen_num += 1

            if not label_dict["s_label"]:
                continue

            new_ex["s_label"] = label_dict["s_label"]
            new_ex["e_label"] = label_dict["e_label"]

        new_examples.append(new_ex)
    logger.info(
        "Delete Overlen Keyphrase (length > 5): %d (overlap / total = %.2f"
        % (overlen_num, float(overlen_num / len(examples) * 100))
        + "%)"
    )
    return new_examples


def bert2chunk_converter(index, ex, tokenizer, mode, max_phrase_words):
    """ convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""
    src_tokens = [BOS_WORD] + ex["tokens"] + [EOS_WORD]
    valid_ids = [0] + ex["valid_mask"] + [0]

    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)
    orig_doc_len = sum(valid_ids)

    if mode == "train":
        s_label = ex["s_label"]
        e_label = ex["e_label"]
        return (
            index,
            src_tensor,
            valid_mask,
            s_label,
            e_label,
            orig_doc_len,
            max_phrase_words,
        )

    else:
        return index, src_tensor, valid_mask, orig_doc_len, max_phrase_words


def batchify_bert2chunk_features_for_train(batch):
    """ train dataloader & eval dataloader ."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    s_label_list = [ex[3] for ex in batch]
    e_label_list = [ex[4] for ex in batch]
    doc_word_lens = [ex[5] for ex in batch]
    max_phrase_words = [ex[6] for ex in batch][0]

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, : d.size(0)].copy_(d)
        input_mask[i, : d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, : v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
    active_mask = torch.LongTensor(len(docs), max_ngram_length).zero_()

    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len

        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            if ngram_len > 0:
                gram_list = [1 for _ in range(ngram_len)] + [0 for _ in range(pad_len)]
            else:
                gram_list = [0 for _ in range(max_word_len - n)]
            batch_mask.extend(gram_list)
        active_mask[batch_i].copy_(torch.LongTensor(batch_mask))

    # -------------------------------------------------------------------
    # [5] label : for n-gram
    # 1. empty label list
    label_list = []
    for _ in range(len(docs)):
        batch_label = []
        for n in range(max_phrase_words):
            batch_label.append(torch.LongTensor([0 for _ in range(max_word_len - n)]))
        label_list.append(batch_label)

    # 2. valid label list
    for batch_i in range(len(docs)):
        for s, e in zip(s_label_list[batch_i], e_label_list[batch_i]):
            gram = e - s
            label_list[batch_i][gram][s] = 1

    # 3. label tensor
    ngram_label = torch.LongTensor(len(docs), max_ngram_length).zero_()
    for batch_i, label in enumerate(label_list):
        ngram_label[batch_i].copy_(torch.cat(label))

    # 4. valid output
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return input_ids, input_mask, valid_ids, active_mask, valid_output, ngram_label, ids


def batchify_bert2chunk_features_for_test(batch):
    """ test dataloader for Dev & Public_Valid."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]
    max_phrase_words = [ex[4] for ex in batch][0]

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, : d.size(0)].copy_(d)
        input_mask[i, : d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, : v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
    active_mask = torch.LongTensor(len(docs), max_ngram_length).zero_()

    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len

        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            if ngram_len > 0:
                gram_list = [1 for _ in range(ngram_len)] + [0 for _ in range(pad_len)]
            else:
                gram_list = [0 for _ in range(max_word_len - n)]
            batch_mask.extend(gram_list)
        active_mask[batch_i].copy_(torch.LongTensor(batch_mask))

    # ---------------------------------------------------------------
    # [5] valid output
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return (
        input_ids,
        input_mask,
        valid_ids,
        active_mask,
        valid_output,
        doc_word_lens,
        ids,
    )
