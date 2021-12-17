from . import dataloader
from .transformers import (
    BertTokenizer,
    RobertaTokenizer,
    BertConfig,
    RobertaConfig,
    DistilBertConfig,
    DistilBertTokenizer,
)
# from transformers import (
#     BertTokenizer,
#     RobertaTokenizer,
#     BertConfig,
#     RobertaConfig,
#     DistilBertConfig,
#     DistilBertTokenizer,
# )

from . import networks
from . import generator
from . import evaluator


from .evaluator import evaluate_openkp, evaluate_kp20k, evaluate_multidata

from .constant import (
    PAD,
    UNK,
    BOS,
    EOS,
    DIGIT,
    PAD_WORD,
    UNK_WORD,
    BOS_WORD,
    EOS_WORD,
    DIGIT_WORD,
    Idx2Tag,
    Tag2Idx,
    IdxTag_Converter,
    Decode_Candidate_Number,
)


tokenizer_class = {
    "bert-base-cased": BertTokenizer,
    "spanbert-base-cased": BertTokenizer,
    "roberta-base": RobertaTokenizer,
    "distilbert-base-multilingual-cased": DistilBertTokenizer,
}

config_class = {
    "bert-base-cased": BertConfig,
    "spanbert-base-cased": BertConfig,
    "roberta-base": RobertaConfig,
    "distilbert-base-multilingual-cased": DistilBertConfig,
}
