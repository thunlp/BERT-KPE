import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from . import loader_utils
from ..constant import BOS_WORD, EOS_WORD, Tag2Idx

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# preprocess label
# ------------------------------------------------------------------------------------------
def get_tag_label(start_end_pos, doc_length):
    # flatten, rank, filter overlap for answer positions
    sorted_positions = loader_utils.flat_rank_pos(start_end_pos)
    filter_positions = loader_utils.strict_filter_overlap(sorted_positions)

    if len(filter_positions) != len(sorted_positions):
        overlap_flag = True
    else:
        overlap_flag = False

    label = [Tag2Idx["O"]] * doc_length
    for s, e in filter_positions:
        if s == e:
            label[s] = Tag2Idx["U"]

        elif (e - s) == 1:
            label[s] = Tag2Idx["B"]
            label[e] = Tag2Idx["E"]

        elif (e - s) >= 2:
            label[s] = Tag2Idx["B"]
            label[e] = Tag2Idx["E"]
            for i in range(s + 1, e):
                label[i] = Tag2Idx["I"]
        else:
            logger.info("ERROR")
            break
    return {"label": label, "overlap_flag": overlap_flag}


def bert2tag_preprocessor(
    examples,
    tokenizer,
    max_token,
    pretrain_model,
    mode,
    max_phrase_words,
    stem_flag=False,
):
    logger.info(
        "start preparing (%s) features for bert2tag (%s) ..." % (mode, pretrain_model)
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
                "doc_length": len(ex["doc_words"]),
            }
            # ------------------------------------------------
            label_dict = get_tag_label(**parameter)
            new_ex["label"] = label_dict["label"][:max_word]
            assert sum(new_ex["valid_mask"]) == len(new_ex["label"])

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
def bert2tag_converter(index, ex, tokenizer, mode, max_phrase_words):
    """ convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""

    src_tokens = [BOS_WORD] + ex["tokens"] + [EOS_WORD]
    valid_ids = [0] + ex["valid_mask"] + [0]

    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)
    orig_doc_len = sum(valid_ids)

    if mode == "train":
        label_tensor = torch.LongTensor(ex["label"])
        return index, src_tensor, valid_mask, orig_doc_len, label_tensor

    else:
        return index, src_tensor, valid_mask, orig_doc_len


def batchify_bert2tag_features_for_train(batch):
    """ train dataloader & eval dataloader ."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]
    label_list = [ex[4] for ex in batch]

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
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, : v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # label tensor
    labels = torch.LongTensor(len(label_list), max_word_len).zero_()
    active_mask = torch.LongTensor(len(label_list), max_word_len).zero_()
    for i, t in enumerate(label_list):
        labels[i, : t.size(0)].copy_(t)
        active_mask[i, : t.size(0)].fill_(1)

    # -------------------------------------------------------------------
    # [6] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return input_ids, input_mask, valid_ids, active_mask, valid_output, labels, ids


def batchify_bert2tag_features_for_test(batch):
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
    # valid length tensor
    active_mask = torch.LongTensor(len(doc_word_lens), max_word_len).zero_()
    for i, l in enumerate(doc_word_lens):
        active_mask[i, :l].fill_(1)

    # -------------------------------------------------------------------
    # [4] Empty Tensor : word-level max_len
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
