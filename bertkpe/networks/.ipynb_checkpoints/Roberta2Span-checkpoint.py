import math
import torch
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import NLLLoss

from ..transformers import BertPreTrainedModel, RobertaModel
# from transformers import BertPreTrainedModel, RobertaModel

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# SelfAtttion Extractor
# -------------------------------------------------------------------------------------------
class SpanAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SpanAttention, self).__init__()

        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, active_mask):
        """hidden_states and active_mask for word_level"""
        attention_mask, tril_mask = self.create_mask(active_mask, hidden_states.size(1))

        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        attention_scores = attention_scores + attention_mask + tril_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs

    def create_mask(self, active_mask, max_len):

        # extended_mask for padding
        extended_active_mask = active_mask[:, None, :]
        extended_active_mask = extended_active_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_active_mask = (1.0 - extended_active_mask) * -10000.0

        full_mask = torch.full([max_len, max_len], -1000.0)
        tril_mask = full_mask.tril_(-1)
        tril_mask = tril_mask.to(next(self.parameters()))
        tril_mask = tril_mask[None, :, :]
        return extended_active_mask, tril_mask


# -------------------------------------------------------------------------------------------
# Inherit BertPreTrainedModel
# -------------------------------------------------------------------------------------------
class RobertaForAttSpanClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForAttSpanClassification, self).__init__(config)
        self.num_labels = config.num_labels

        # RobertaModel
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.self_att_classifier = SpanAttention(config.hidden_size)

        self.init_weights()


# -------------------------------------------------------------------------------------------
# RobertaForAttSpanExtractor
# -------------------------------------------------------------------------------------------
class RobertaForAttSpanExtractor(RobertaForAttSpanClassification):
    def forward(
        self,
        input_ids,
        attention_mask,
        valid_ids,
        valid_output,
        active_mask,
        s_label=None,
        e_label=None,
        end_mask=None,
    ):

        # --------------------------------------------------------------------------------
        # Bert Embedding Outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        # --------------------------------------------------------------------------------
        # Valid Outputs : get first token vector
        batch_size = sequence_output.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_ids[i]).item()

            vectors = sequence_output[i][valid_ids[i] == 1]
            valid_output[i, :valid_num].copy_(vectors)

        # --------------------------------------------------------------------------------
        # Dropout
        sequence_output = self.dropout(
            valid_output
        )  # shape = (batch_size * max_word_length * 768)

        # start softmax logit
        s_logits = self.classifier(sequence_output)
        s_logits = F.softmax(s_logits, dim=-1)

        # end softmax logit
        e_logits = self.self_att_classifier(
            hidden_states=sequence_output, active_mask=active_mask
        )  # shape = (batch_size, max_word_len, max_word_len)

        s_active_loss = active_mask.view(-1) == 1  # [False, True, ...]
        s_active_logits = s_logits.view(-1, self.num_labels)[s_active_loss]  # False

        if (s_label is not None) and (e_label is not None):

            loss_fct = NLLLoss()

            # --------------------------------------------------------
            # Start Loss : log
            s_active_logits = torch.log(s_active_logits + 1e-16)
            # Start Loss : activate label
            s_active_labels = s_label.view(-1)[s_active_loss]
            # Start Loss : final start loss
            start_loss = loss_fct(s_active_logits, s_active_labels)

            # --------------------------------------------------------
            # End Loss : log (+ 1e-16 prevent -inf)
            e_logits = torch.log(e_logits + 1e-16)
            # End Loss : activate end loss from s_label
            e_active_loss = s_label.view(-1) == 1
            e_active_logits = e_logits.view(-1, e_logits.shape[1])[e_active_loss]

            # --------------------------------------------------------
            # End Loss :  activate end label
            e_label_valid_ids = end_mask.view(-1) == 1
            e_activate_labels = e_label.view(-1)[e_label_valid_ids]
            end_loss = loss_fct(e_active_logits, e_activate_labels)

            # --------------------------------------------------------
            # total loss
            total_loss = start_loss + end_loss  # (start_loss + end_loss) / 2
            return total_loss
        else:
            e_active_logits = e_logits.view(-1, e_logits.shape[1])[s_active_loss]
            s_active_logits = s_active_logits[:, 1]
            return s_active_logits, e_active_logits
