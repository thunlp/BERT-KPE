import math
import torch
import logging
import traceback
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss

from ..transformers import BertPreTrainedModel, RobertaModel
# from transformers import BertPreTrainedModel, RobertaModel

from itertools import repeat
import collections.abc as container_abcs

logger = logging.getLogger()

# -------------------------------------------------------------------------------------------
# Modified CNN
# -------------------------------------------------------------------------------------------


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)


class _ConvNd(nn.Module):

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
    ):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):

        input = input.transpose(1, 2)

        if self.padding_mode == "circular":
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(
                F.pad(input, expanded_padding, mode="circular"),
                self.weight,
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )

        output = F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output = output.transpose(1, 2)
        return output


# -------------------------------------------------------------------------------------------
# CnnGram Extractor
# -------------------------------------------------------------------------------------------


class CNNGramer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.cnn_1 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )

        self.cnn_2 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=2
        )

        self.cnn_3 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=3
        )

        self.cnn_4 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=4
        )

        self.cnn_5 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=5
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        gram_1 = self.cnn_1(x)
        gram_2 = self.cnn_2(x)
        gram_3 = self.cnn_3(x)
        gram_4 = self.cnn_4(x)
        gram_5 = self.cnn_5(x)

        outputs = torch.cat(
            [
                gram_1.transpose(1, 2),
                gram_2.transpose(1, 2),
                gram_3.transpose(1, 2),
                gram_4.transpose(1, 2),
                gram_5.transpose(1, 2),
            ],
            1,
        )
        return outputs


# -------------------------------------------------------------------------------------------
# Inherit BertPreTrainedModel
# -------------------------------------------------------------------------------------------
class RobertaForCnnGramClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForCnnGramClassification, self).__init__(config)

        cnn_output_size = 512

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)

        self.cnn2gram = CNNGramer(
            input_size=config.hidden_size, hidden_size=cnn_output_size
        )

        self.classifier = nn.Linear(cnn_output_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()


# -------------------------------------------------------------------------------------------
# RobertaForCnnGramExtractor
# -------------------------------------------------------------------------------------------


class RobertaForCnnGramExtractor(RobertaForCnnGramClassification):
    def forward(
        self,
        input_ids,
        attention_mask,
        valid_ids,
        active_mask,
        valid_output,
        labels=None,
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
        sequence_output = self.dropout(valid_output)

        # --------------------------------------------------------------------------------
        # CNN Outputs
        cnn_outputs = self.cnn2gram(
            sequence_output
        )  # shape = (batch_size, max_gram_num, 512)

        # classifier (512, 2)
        logits = self.classifier(cnn_outputs)

        # activate loss
        active_loss = (
            active_mask.view(-1) == 1
        )  # [False, True, ...] # batch_size * max_len
        active_logits = logits.view(-1, self.num_labels)[active_loss]  # False

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return active_logits
