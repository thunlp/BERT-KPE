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
def get_span_label(start_end_pos, max_doc_word):
    # flatten, rank, filter overlap for answer positions
    sorted_positions = loader_utils.flat_rank_pos(start_end_pos)
    filter_positions = loader_utils.loose_filter_overlap(sorted_positions)

    if len(filter_positions) != len(sorted_positions):
        overlap_flag = True
    else:
        overlap_flag = False

    s_label = [0] * max_doc_word
    e_label = []
    for s, e in filter_positions:
        if (s <= e) and (e < max_doc_word):
            s_label[s] = 1
            e_label.append(e)
        else:
            continue

    if (len(e_label) > 0) and (sum(s_label) == len(e_label)):
        return {"s_label": s_label, "e_label": e_label, "overlap_flag": overlap_flag}
    else:
        return {"s_label": None, "e_label": None, "overlap_flag": overlap_flag}


def bert2span_preprocessor(
    examples,
    tokenizer,
    max_token,
    pretrain_model,
    mode,
    max_phrase_words,
    stem_flag=False,
):
    logger.info(
        "start preparing (%s) features for bert2span (%s) ..." % (mode, pretrain_model)
    )
    overlap_num = 0
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
                "start_end_pos": ex["start_end_pos"],
                "max_doc_word": len(new_ex["doc_words"]),
            }

            label_dict = get_span_label(**parameter)
            if not label_dict["s_label"]:
                continue

            new_ex["s_label"] = label_dict["s_label"]
            new_ex["e_label"] = label_dict["e_label"]

            assert sum(new_ex["valid_mask"]) == len(new_ex["s_label"])
            if label_dict["overlap_flag"]:
                overlap_num += 1
        new_examples.append(new_ex)

    logger.info(
        "Delete Overlap Keyphrase : %d (overlap / total = %.2f"
        % (overlap_num, float(overlap_num / len(examples) * 100))
        + "%)"
    )
    return new_examples


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# batch batchfy
def bert2span_converter(index, ex, tokenizer, mode, max_phrase_words):
    """ convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""

    src_tokens = [BOS_WORD] + ex["tokens"] + [EOS_WORD]
    valid_ids = [0] + ex["valid_mask"] + [0]

    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)
    orig_doc_len = sum(valid_ids)

    if mode == "train":
        s_label_tensor = torch.LongTensor(ex["s_label"])
        e_label_tensor = torch.LongTensor(ex["e_label"])
        return (
            index,
            src_tensor,
            valid_mask,
            orig_doc_len,
            s_label_tensor,
            e_label_tensor,
        )
    else:
        return index, src_tensor, valid_mask, orig_doc_len


def batchify_bert2span_features_for_train(batch):
    """ train dataloader & eval dataloader ."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]
    s_label_list = [ex[4] for ex in batch]
    e_label_list = [ex[5] for ex in batch]

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1][2]src tokens tensor
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
    # [4] start label [5] active_mask
    s_label = torch.LongTensor(len(s_label_list), max_word_len).zero_()
    active_mask = torch.LongTensor(len(s_label_list), max_word_len).zero_()
    for i, s in enumerate(s_label_list):
        s_label[i, : s.size(0)].copy_(s)
        active_mask[i, : s.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # [6] end label [7] end_mask
    e_label_max_length = max([e.size(0) for e in e_label_list])
    e_label = torch.LongTensor(len(e_label_list), e_label_max_length).zero_()
    end_mask = torch.LongTensor(len(e_label_list), e_label_max_length).zero_()
    for i, e in enumerate(e_label_list):
        if e.size(0) <= 0:
            continue
        e_label[i, : e.size(0)].copy_(e)
        end_mask[i, : e.size(0)].fill_(1)

    # -------------------------------------------------------------------
    # [8] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return (
        input_ids,
        input_mask,
        valid_ids,
        valid_output,
        active_mask,
        s_label,
        e_label,
        end_mask,
        ids,
    )


def batchify_bert2span_features_for_test(batch):
    """ test dataloader for Dev & Public_Valid."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1][2]src tokens tensor
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
    # [4] valid length tensor
    active_mask = torch.LongTensor(len(doc_word_lens), max_word_len).zero_()
    for i, l in enumerate(doc_word_lens):
        active_mask[i, :l].fill_(1)

    # -------------------------------------------------------------------
    # [5] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return (
        input_ids,
        input_mask,
        valid_ids,
        valid_output,
        active_mask,
        doc_word_lens,
        ids,
    )
