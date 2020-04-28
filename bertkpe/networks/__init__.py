def get_class(args):
    
    # Bert2Span
    if args.model_class == 'bert2span' and args.pretrain_model_type in ['bert-base-cased', 'spanbert-base-cased']:
        return BertForAttSpanExtractor
    elif args.model_class == 'bert2span' and args.pretrain_model_type == 'roberta-base':
        return RobertaForAttSpanExtractor
    
    # Bert2Tag
    elif args.model_class == 'bert2tag' and args.pretrain_model_type in ['bert-base-cased', 'spanbert-base-cased']:
        return BertForSeqTagging
    elif args.model_class == 'bert2tag' and args.pretrain_model_type == 'roberta-base':
        return RobertaForSeqTagging
    
    # Bert2Chunk
    elif args.model_class == 'bert2chunk' and args.pretrain_model_type in ['bert-base-cased', 'spanbert-base-cased']:
        return BertForCnnGramExtractor
    elif args.model_class == 'bert2chunk' and args.pretrain_model_type == 'roberta-base':
        return RobertaForCnnGramExtractor
    
    # Bert2Rank
    elif args.model_class == 'bert2rank' and args.pretrain_model_type in ['bert-base-cased', 'spanbert-base-cased']:
        return BertForTFRanking
    elif args.model_class == 'bert2rank' and args.pretrain_model_type == 'roberta-base':
        return RobertaForTFRanking
    
    # Bert2Joint
    elif args.model_class == 'bert2joint' and args.pretrain_model_type in ['bert-base-cased', 'spanbert-base-cased']:
        return BertForChunkTFRanking
    elif args.model_class == 'bert2joint' and args.pretrain_model_type == 'roberta-base':
        return RobertaForChunkTFRanking
    
    raise RuntimeError('Invalid retriever class: %s' % name)


# bert2span
from .Bert2Span import BertForAttSpanExtractor
from .Roberta2Span import RobertaForAttSpanExtractor

# bert2tag
from .Bert2Tag import BertForSeqTagging
from .Roberta2Tag import RobertaForSeqTagging

# bert2chunk
from .Bert2Chunk import BertForCnnGramExtractor
from .Roberta2Chunk import RobertaForCnnGramExtractor

# bert2rank
from .Bert2Rank import BertForTFRanking
from .Roberta2Rank import RobertaForTFRanking

# bert2joint
from .Bert2Joint import BertForChunkTFRanking
from .Roberta2Joint import RobertaForChunkTFRanking