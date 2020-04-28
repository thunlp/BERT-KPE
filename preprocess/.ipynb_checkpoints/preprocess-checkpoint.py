import os
import re
import sys
import json
import time
import torch
import codecs
import pickle
import logging
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm
import prepro_utils

dataset_dict = {'openkp':[('EvalPublic', 'eval'), ('Dev', 'dev'), ('Train', 'train')], 
                'kp20k':[('testing', 'eval'), ('validation', 'dev'), ('training', 'train')]}

logger = logging.getLogger()
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# config param
def add_preprocess_opts(parser):
    
    parser.add_argument("--dataset_class", type=str, choices=['openkp', 'kp20k'],
                       help="Select dataset to be preprocessed. ")
    parser.add_argument('--source_dataset_dir', type=str, required=True,
                        help="The path to the source dataset (raw json).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="The path to save preprocess data")
    # ------------------------------------------------------------------
    # specific for kp20k
    parser.add_argument('-max_src_seq_length', type=int, default=300,
                        help="Maximum document sequence length")
    parser.add_argument('-min_src_seq_length', type=int, default=20,
                        help="Minimum document sequence length")
    parser.add_argument('-max_trg_seq_length', type=int, default=6,
                        help="Maximum keyphrases sequence length to keep.")
    parser.add_argument('-min_trg_seq_length', type=int, default=0,
                        help="Minimun keyphrases sequence length to keep.")

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# logging file
def set_logger(log_file):
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 

    logfile = logging.FileHandler(log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# original dataset loader
def openkp_loader(mode, source_dataset_dir):
    ''' load source OpenKP dataset :'url', 'VDOM', 'text', 'KeyPhrases' '''
    
    logger.info("start loading %s data ..." % mode)
    source_path = os.path.join(source_dataset_dir, 'OpenKP%s.jsonl' % mode)
    data_pairs = []
    with codecs.open(source_path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(tqdm(corpus_file)):
            json_ = json.loads(line)
            data_pairs.append(json_)
    return data_pairs


def kp20k_loader(mode, source_dataset_dir, 
                 src_fields = ['title', 'abstract'], 
                 trg_fields = ['keyword'], trg_delimiter=';'):
    
    ''' load source Kp20k dataset :'title', 'abstract', 'keyword' 
    return : tuple : src_string, trg_string'''
    
    logger.info("start loading %s data ..." % mode)
    source_path = os.path.join(source_dataset_dir, 'kp20k_%s.json' % mode)
    
    data_pairs = []
    with codecs.open(source_path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(tqdm(corpus_file)):
            json_ = json.loads(line)
            
            trg_strs = []
            src_str = '.'.join([json_[f] for f in src_fields])
            [trg_strs.extend(re.split(trg_delimiter, json_[f])) for f in trg_fields]
            data_pairs.append((src_str, trg_strs))
    return data_pairs

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# first stage preprocess
def openkp_refactor(examples, mode):
    logger.info('strat refactor openkp %s data ...'%mode)
    
    have_phrase = True
    if mode == 'EvalPublic':
        have_phrase = False
    
    return_pairs = []
    for idx, ex in enumerate(tqdm(examples)):
        doc_words, word2block, block_features = prepro_utils.refactor_text_vdom(text=ex['text'], 
                                                                                VDOM=ex['VDOM'])
        data = {}
        data['url'] = ex['url']
        data['doc_words'] = doc_words
        data['word2block'] = word2block
        data['block_features'] = block_features
        if have_phrase:
            keyphrases = prepro_utils.clean_phrase(ex['KeyPhrases'])
            data['keyphrases'] = keyphrases
            
        return_pairs.append(data)
    return return_pairs


def kp20k_refactor(src_trgs_pairs, mode, valid_check=True):
    logger.info("start refactor kp20k %s data ..." % mode)

    def tokenize_fn(text):
        '''
        The tokenizer used in Meng et al. ACL 2017
        parse the feed-in text, filtering and tokenization
        keep [_<>,\(\)\.\'%], replace digits to 'DIGIT', split by [^a-zA-Z0-9_<>,\(\)\.\'%]
        :return: a list of tokens
        '''
        DIGIT = 'DIGIT'
        # remove line breakers
        text = re.sub(r'[\r\n\t]', ' ', text)
        # pad spaces to the left and right of special punctuations
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
        # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))
        
        # replace the digit terms with DIGIT
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

        return tokens

    # ---------------------------------------------------------------------------------------
    return_pairs = []
    for idx, (src, trgs) in enumerate(tqdm(src_trgs_pairs)):
        src_filter_flag = False

        src_tokens = tokenize_fn(src)

        # FILTER 3.1: if length of src exceeds limit, discard
        if opt.max_src_seq_length and len(src_tokens) > opt.max_src_seq_length:
            src_filter_flag = True
        if opt.min_src_seq_length and len(src_tokens) < opt.min_src_seq_length:
            src_filter_flag = True

        if valid_check and src_filter_flag:
            continue

        trgs_tokens = []
        for trg in trgs:
            trg_filter_flag = False
            trg = trg.lower()

            # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
            trg = re.sub(r'\(.*?\)', '', trg)
            trg = re.sub(r'\[.*?\]', '', trg)
            trg = re.sub(r'\{.*?\}', '', trg)

            # FILTER 2: ingore all the phrases that contains strange punctuations, very DIRTY data!
            puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', trg)

            trg_tokens = tokenize_fn(trg)

            if len(puncts) > 0:
                continue
                # Find punctuations in keyword: %s' % trg

            # FILTER 3.2: if length of trg exceeds limit, discard
            if opt.max_trg_seq_length and len(trg_tokens) > opt.max_trg_seq_length:
                trg_filter_flag = True
            if opt.min_trg_seq_length and len(trg_tokens) < opt.min_trg_seq_length:
                trg_filter_flag = True

            filtered_by_heuristic_rule = False

            # FILTER 4: check the quality of long keyphrases (>5 words) with a heuristic rule
            if len(trg_tokens) > 5:
                trg_set = set(trg_tokens)
                if len(trg_set) * 2 < len(trg_tokens):
                    filtered_by_heuristic_rule = True

            if valid_check and (trg_filter_flag or filtered_by_heuristic_rule):
                continue
                # length of src/trg exceeds limit: len(src)=%d, len(trg)=%d
                # if filtered_by_heuristic_rule:print('INVALID by heuristic_rule')
                # else:VALID by heuristic_rule

            # FILTER 5: filter keywords like primary 75v05;secondary 76m10;65n30
            if valid_check and (len(trg_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', 
                                                                 trg_tokens[0].strip())) or (len(trg_tokens) > 1 and re.match(r'\d\d\w\d\d', trg_tokens[1].strip())):
                continue
                # print('Find dirty keyword of type \d\d[a-z]\d\d: %s' % trg)

            trgs_tokens.append(trg_tokens)

        # ignore the examples that have zero valid targets, for training they are no helpful
        if valid_check and len(trgs_tokens) == 0:
            continue

        return_pairs.append({'doc_words': src_tokens, 'keyphrases':trgs_tokens})
        # return_pairs.append((src_tokens, trgs_tokens))
        
    return return_pairs


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# filter absent keyphrases
def filter_openkp_absent(examples):
    logger.info('strat filter absent keyphrases ...')
    data_list = []
    null_urls, absent_urls = [], []
    for idx, ex in enumerate(tqdm(examples)):
        
        lower_words = [t.lower() for t in ex['doc_words']]
        present_phrases = prepro_utils.find_answer(document=lower_words, 
                                                   answers=ex['keyphrases'])
        if present_phrases is None:
            null_urls.append(ex['url'])
            continue
        if len(present_phrases['keyphrases']) != len(ex['keyphrases']):
            absent_urls.append(ex['url'])
            
        data = {}
        data['url'] = ex['url']
        data['doc_words'] = ex['doc_words']
        data['word2block'] = ex['word2block']
        data['block_features'] = ex['block_features']
        data['keyphrases'] = ex['keyphrases']
        # -------------------------------------------------------
        # new added
        data['start_end_pos'] = present_phrases['start_end_pos'] 
        data['present_keyphrases'] = present_phrases['keyphrases']
        data_list.append(data)
        
    logger.info('Null : number = {} , URL = {} '.format(len(null_urls), null_urls))
    logger.info('Absent : number = {} , URL = {} '.format(len(absent_urls), absent_urls))
    return data_list


def filter_kp20k_absent(examples):
    logger.info('strat filter absent keyphrases for KP20k...')
    data_list = []
    
    null_ids, absent_ids = 0, 0
    
    url = 0
    for idx, ex in enumerate(tqdm(examples)):
        
        lower_words = [t.lower() for t in ex['doc_words']]
        present_phrases = prepro_utils.find_stem_answer(word_list=lower_words, ans_list=ex['keyphrases'])
        if present_phrases is None:
            null_ids += 1
            continue
        if len(present_phrases['keyphrases']) != len(ex['keyphrases']):
            absent_ids += 1
            
        data = {}
        data['url'] = url
        data['doc_words'] = ex['doc_words']
        data['keyphrases'] = present_phrases['keyphrases']
        data['start_end_pos'] = present_phrases['start_end_pos']
        
        data_list.append(data)
        url += 1
        
    logger.info('Null : number = {} '.format(null_ids))
    logger.info('Absent : number = {} '.format(absent_ids))
    return data_list


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# saver
def save_ground_truths(examples, filename, kp_key):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for ex in tqdm(examples):
            data = {}
            data['url'] = ex['url']
            data['KeyPhrases'] = ex[kp_key]
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save reference to %s'%filename)

    
def save_preprocess_data(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    logger.info("Success save file to %s \n" %filename)
    
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# main function
def main_preprocess(opt, input_mode, save_mode):
    
    # load source dataset
    source_data = data_loader[opt.dataset_class](input_mode, opt.source_dataset_dir)
    logger.info("success loaded %s %s data : %d " % (opt.dataset_class, input_mode, len(source_data)))

    # refactor source data
    refactor_data = data_refactor[opt.dataset_class](source_data, input_mode)
    logger.info('success refactor %s source data !' % input_mode)

    # filter absent keyphrases (kp20k use stem)
    if opt.dataset_class == 'openkp' and input_mode == 'EvalPublic':
        feed_data = refactor_data
    else:
        feed_data = absent_filter[opt.dataset_class](refactor_data)
        logger.info('success obtain %s present keyphrase : %d (filter out : %d)'
                    %(input_mode, len(feed_data), (len(refactor_data) - len(feed_data))))
        
    # save ground-truth
    gt_filename = os.path.join(opt.output_path, "%s.%s_candidate.json"%(opt.dataset_class, save_mode))
    if opt.dataset_class == 'kp20k' and save_mode != 'train':
        save_ground_truths(feed_data, gt_filename, kp_key='keyphrases')
        
    # openkp dev ground-truth keyphrases
    if opt.dataset_class == 'openkp' and save_mode == 'dev':
        save_ground_truths(source_data, gt_filename, kp_key='KeyPhrases')
    
    # -------------------------------------------------------------------------------------
    # save preprocess features
    save_filename = os.path.join(opt.output_path, "%s.%s.json" % (opt.dataset_class, save_mode))
    save_preprocess_data(feed_data, save_filename)
    logger.info("success saved %s %s data: %d to %s" 
                %(opt.dataset_class, save_mode, len(feed_data), save_filename))
    

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # preprocess selector
    data_loader = {'openkp':openkp_loader, 'kp20k':kp20k_loader}
    data_refactor = {'openkp':openkp_refactor, 'kp20k':kp20k_refactor}
    absent_filter = {'openkp':filter_openkp_absent, 'kp20k':filter_kp20k_absent}

    t0 = time.time()
    parser = argparse.ArgumentParser(description='preprocess_for_kpe.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add options
    add_preprocess_opts(parser)
    opt = parser.parse_args()

    # option folder
    opt.source_dataset_dir = os.path.join(opt.source_dataset_dir, opt.dataset_class)
    if not os.path.exists(opt.source_dataset_dir):
        logger.info("don't exist the source dataset dir: %s" % opt.source_dataset_dir)

    opt.output_path = os.path.join(opt.output_path, opt.dataset_class)
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
        logger.info("don't exist this output dir, remake a new to : %s" % opt.output_path)

    # set logger
    opt.log_file = os.path.join(opt.output_path, 'prepro_logging.txt')
    set_logger(opt.log_file)

    # start preprocess ...
    mode_dir = dataset_dict[opt.dataset_class]
    for input_mode, save_mode in mode_dir:
        main_preprocess(opt, input_mode, save_mode)