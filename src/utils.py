import os
import time
import json
import torch
import string
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
logger = logging.getLogger(__name__)
from Constants import BOS_WORD, EOS_WORD, DIGIT_WORD


# -------------------------------------------------------------------------------------------
# load datasets or preprocess features
# -------------------------------------------------------------------------------------------
def read_openkp_examples(args, tokenizer):
    ''' load preprocess cached_features files. '''            
    if not os.listdir(args.preprocess_folder):
        logger.info('Error : not found %s' % args.preprocess_folder)

    if args.run_mode == 'train':
        mode_dict = {'train':[], 'valid':[]}
    elif args.run_mode == 'generate':
        mode_dict = {'eval_public':[], 'valid':[]}
    else:
        raise Exception("Invalid run mode %s!" % args.run_mode)
        
    for mode in mode_dict:
        filename = os.path.join(args.preprocess_folder, "openkp.%s.json" % mode)
        logger.info("start loading openkp %s data ..." %mode)
        with open(filename, "r", encoding="utf-8") as f:
            mode_dict[mode] = json.load(f)
        f.close()
        logger.info("success loaded openkp %s data : %d " %(mode, len(mode_dict[mode])))
    return mode_dict


# -------------------------------------------------------------------------------------------
# build dataset and dataloader
# -------------------------------------------------------------------------------------------        
class build_openkp_dataset(Dataset):
    ''' build datasets for train & eval '''
    def __init__(self, args, examples, tokenizer, converter, shuffle=False):
        
        self.run_mode = args.run_mode
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src_len = args.max_src_len
        self.converter = converter
        if shuffle:
            random.seed(args.seed)
            random.shuffle(self.examples)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return convert_examples_to_features(index, self.examples[index], self.tokenizer, 
                                            self.converter, self.max_src_len, self.run_mode)
                
        
def convert_examples_to_features(index, ex, tokenizer, converter, max_src_len, run_mode):
    ''' convert each batch data to tensor ; add [CLS] [SEP] tokens ; cut over 512 tokens.'''
    
    src_tokens = [BOS_WORD] + ex['tokens'][:max_src_len] + [EOS_WORD] # max_src_len = 510
    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    
    valid_ids = [0] + ex['valid_mask'][:max_src_len] + [0]
    valid_mask = torch.LongTensor(valid_ids)

    if run_mode == 'train':
        if len(ex['tokens']) < max_src_len:
            orig_max_src_len = max_src_len
        else:
            orig_max_src_len = ex['tok_to_orig_index'][max_src_len-1] + 1
        label_tensor = torch.LongTensor(converter.convert_tag2idx(ex['label'][:orig_max_src_len]))
        return index, src_tensor, valid_mask, label_tensor
    
    elif run_mode == 'generate':
        valid_orig_doc_len = sum(valid_ids)
        return index, src_tensor, valid_mask, valid_orig_doc_len
    
    else:
        logger.info('not the mode : %s'% run_mode)

    
def batchify_features_for_train_eval(batch):
    ''' train dataloader & eval dataloader .'''
    
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    label_list = [ex[3] for ex in batch]

    # ---------------------------------------------------------------
    # src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)
        
    # ---------------------------------------------------------------
    # label tensor
    labels = torch.LongTensor(len(label_list), doc_max_length).zero_()
    active_mask = torch.LongTensor(len(label_list), doc_max_length).zero_()
    for i, t in enumerate(label_list):
        labels[i, :t.size(0)].copy_(t)
        active_mask[i, :t.size(0)].fill_(1)
        
    assert input_ids.size() == valid_ids.size() == labels.size()
    return input_ids, input_mask, valid_ids, active_mask, labels, ids


def batchify_features_for_test(batch):
    ''' test dataloader for Dev & Public_Valid.'''
    
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    valid_lens = [ex[3] for ex in batch]

    # ---------------------------------------------------------------
    # src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)
        
    # ---------------------------------------------------------------
    # valid length tensor
    active_mask = torch.LongTensor(len(valid_lens), doc_max_length).zero_()
    for i, l in enumerate(valid_lens):
        active_mask[i, :l].fill_(1)

    assert input_ids.size() == valid_ids.size() == active_mask.size()
    return input_ids, input_mask, valid_ids, active_mask, valid_lens, ids


# -------------------------------------------------------------------------------------------
# other utils fucntions
# -------------------------------------------------------------------------------------------     
def override_args(old_args, new_args):
    ''' cover old args to new args, log which args has been changed.'''
    
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            logger.info('Overriding saved %s: %s --> %s' %
                        (k, old_args[k], new_args[k]))
            old_args[k] = new_args[k]
    return argparse.Namespace(**old_args) 
        

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
    
