import os
import json
import codecs
import logging
import unicodedata
from tqdm import tqdm
from ..constant import UNK_WORD, BOS_WORD, EOS_WORD
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset
stemmer = PorterStemmer()

from .bert2span_dataloader import (bert2span_preprocessor, bert2span_converter)
from .bert2tag_dataloader import (bert2tag_preprocessor, bert2tag_converter)
from .bert2chunk_dataloader import (bert2chunk_preprocessor, bert2chunk_converter)

from .bert2rank_dataloader import (bert2rank_preprocessor, bert2rank_converter)
from .bert2joint_dataloader import (bert2joint_preprocessor, bert2joint_converter)


example_preprocessor = {'bert2span':bert2span_preprocessor, 
                        'bert2tag':bert2tag_preprocessor, 
                        'bert2chunk':bert2chunk_preprocessor, 
                        'bert2rank':bert2rank_preprocessor, 
                        'bert2joint':bert2joint_preprocessor}

feature_converter = {'bert2span':bert2span_converter, 
                     'bert2tag':bert2tag_converter, 
                     'bert2chunk':bert2chunk_converter, 
                     'bert2rank':bert2rank_converter, 
                     'bert2joint':bert2joint_converter}


logger = logging.getLogger()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# load & save source dataset
def load_dataset(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with codecs.open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            jsonl = json.loads(line)
            data_list.append(jsonl)
    logger.info('success load %d data'%len(data_list))
    return data_list


def save_dataset(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    logger.info("Success save %d data to %s" %(len(data_list), filename))
    
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# build dataset
class build_dataset(Dataset):
    ''' build datasets for train & eval '''
    def __init__(self, args, tokenizer, mode):
        
        pretrain_model = 'bert' if 'roberta' not in args.pretrain_model_type else 'roberta'
        # --------------------------------------------------------------------------------------------
        # try to reload cached features
        try:
            cached_examples = reload_cached_features(**{'cached_features_dir':args.cached_features_dir,
                                                        'model_class':args.model_class,
                                                        'dataset_class':args.dataset_class, 
                                                        'pretrain_model':pretrain_model,
                                                        'mode':mode})
        # --------------------------------------------------------------------------------------------
        # restart preprocessing features
        except:
            logger.info("start loading source %s %s data ..." %(args.dataset_class, mode))            
            examples = load_dataset(os.path.join(args.preprocess_folder, "%s.%s.json" % (args.dataset_class, mode)))
            cached_examples = example_preprocessor[args.model_class](**{'examples':examples, 
                                                                        'tokenizer':tokenizer, 
                                                                        'max_token':args.max_token,
                                                                        'pretrain_model':pretrain_model, 
                                                                        'mode':mode,
                                                                        'max_phrase_words':args.max_phrase_words,
                                                                        'stem_flag':True if args.dataset_class == 'kp20k' else False})
            if args.local_rank in [-1, 0]:
                save_cached_features(**{'cached_examples':cached_examples, 
                                        'cached_features_dir':args.cached_features_dir, 
                                        'model_class':args.model_class, 
                                        'dataset_class':args.dataset_class, 
                                        'pretrain_model':pretrain_model,
                                        'mode':mode})
        # --------------------------------------------------------------------------------------------
        self.mode = mode
        self.tokenizer = tokenizer
        self.examples = cached_examples
        self.model_class = args.model_class
        self.max_phrase_words = args.max_phrase_words
        # --------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return feature_converter[self.model_class](index, self.examples[index], 
                                                   self.tokenizer, self.mode, self.max_phrase_words)
    

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# pre-trained model tokenize
def tokenize_for_bert(doc_words, tokenizer):        
    valid_mask = []
    all_doc_tokens = []
    tok_to_orig_index = []
    orig_to_tok_index = [] 
    for (i, token) in enumerate(doc_words):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        if len(sub_tokens) < 1:
            sub_tokens = [UNK_WORD]
        for num, sub_token in enumerate(sub_tokens):
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            if num == 0:
                valid_mask.append(1)
            else:
                valid_mask.append(0)
    return {'tokens':all_doc_tokens, 
            'valid_mask':valid_mask, 
            'tok_to_orig_index':tok_to_orig_index, 
            'orig_to_tok_index':orig_to_tok_index}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# load & save cached features
def reload_cached_features(cached_features_dir, model_class, 
                          dataset_class, pretrain_model, mode):
    logger.info("start reloading:  %s (%s) for %s (%s) cached features ..." 
                %(model_class, pretrain_model, dataset_class, mode))
    filename = os.path.join(cached_features_dir, "cached.%s.%s.%s.%s.json" 
                            %(model_class, pretrain_model, dataset_class, mode))
    
    examples = load_dataset(filename)
    return examples


def save_cached_features(cached_examples, cached_features_dir, 
                        model_class, dataset_class, pretrain_model, mode):
    logger.info("start saving:  %s (%s) for %s (%s) cached features ..." 
                %(model_class, pretrain_model, dataset_class, mode))
    if not os.path.exists(cached_features_dir):
        os.mkdir(cached_features_dir)
        
    save_filename = os.path.join(cached_features_dir, "cached.%s.%s.%s.%s.json" 
                            %(model_class, pretrain_model, dataset_class, mode))
    save_dataset(data_list=cached_examples, filename=save_filename)


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# fucntions for converting labels
def flat_rank_pos(start_end_pos):
    flatten_postions = [pos for poses in start_end_pos for pos in poses]
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    return sorted_positions


def strict_filter_overlap(positions):
    '''delete overlap keyphrase positions. '''
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def loose_filter_overlap(positions):
    '''delete overlap keyphrase positions. '''
    previous_s = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if previous_s == s:
            continue
        elif previous_s < s:
            filter_positions.append(positions[i])
            previous_s = s
        else:
            logger.info('Error! previous start large than new start')
    return filter_positions


def limit_phrase_length(positions, max_phrase_words):
    filter_positions = [pos for pos in positions if (pos[1]-pos[0]+1) <= max_phrase_words]
    return filter_positions


# Delete Over Scope keyphrase position (token_len > 510) and phrase_length > 5
def limit_scope_length(start_end_pos, valid_length, max_phrase_words):
    """filter out positions over scope & phase_length > 5"""
    filter_positions = []
    for positions in start_end_pos:
        _filter_position = [pos for pos in positions \
                            if pos[1] < valid_length and (pos[1]-pos[0]+1) <= max_phrase_words]
        if len(_filter_position) > 0:
            filter_positions.append(_filter_position)
    return filter_positions



def stemming(phrase):
    norm_chars = unicodedata.normalize('NFD', phrase)
    stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
    return norm_chars, stem_chars



def whether_stem_existing(gram, phrase2index, tot_phrase_list):
    """If :
       unicoding(gram) and stemming(gram) not in phrase2index, 
       Return : not_exist_flag
       Else :
       Return : index already in phrase2index.
    """
    norm_gram, stem_gram = stemming(gram)
    if norm_gram in phrase2index:
        index = phrase2index[norm_gram]
        phrase2index[stem_gram] = index
        return index

    elif stem_gram in phrase2index:
        index = phrase2index[stem_gram]
        phrase2index[norm_gram] = index
        return index

    else:
        index = len(tot_phrase_list)
        phrase2index[norm_gram] = index
        phrase2index[stem_gram] = index
        tot_phrase_list.append(gram)
        return index
    
    
def whether_existing(gram, phrase2index, tot_phrase_list):
    """If :
       gram not in phrase2index, 
       Return : not_exist_flag
       Else :
       Return : index already in phrase2index.
    """
    if gram in phrase2index:
        index = phrase2index[gram]
        return index
    else:
        index = len(tot_phrase_list)
        phrase2index[gram] = index
        tot_phrase_list.append(gram)
        return index
    