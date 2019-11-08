import re
import math
import string
import logging
import collections 
import unicodedata
import numpy as np

import prepro_utils
logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# OpenKP official evaluation script 

def get_subset_grountruths(cut_tokens, cut_labels, full_phrase_number):
    '''get groundtruth keyphrases from cut tokens .'''
    chunk_postions, chunk_scores = prepro_utils.get_chunk_positions(cut_labels)
    groundtruths, _ = decode_pos2phrase(cut_tokens, chunk_postions, chunk_scores, full_phrase_number)
    return groundtruths


def decode_pos2phrase(orig_tokens, sort_positions, sort_scores, n_best=None):
    
    phrase_set = set()
    n_best_phrases = []
    n_best_scores = []
    for i, (pred_s, pred_e) in enumerate(sort_positions):
            
        # don't filter punctution '.' ',' ...
        final_text = ' '.join(orig_tokens[pred_s:pred_e+1])
    
        # remove dupilicate
        if final_text.lower() in phrase_set:
            continue
            
        n_best_phrases.append(final_text)
        n_best_scores.append(sort_scores[i])
        
        phrase_set.add(final_text.lower())

    assert len(n_best_phrases) == len(n_best_scores)
    
    if n_best:
        return n_best_phrases[:n_best], n_best_scores[:n_best]
    return n_best_phrases, n_best_scores

    

def decode_candidates(orig_tokens, scores, indexs, converter, n_best=None):
    '''
    Input :  
        orig_tokens : word's tokens before wordpiece
        scores : a list each tag prediction score
        indexs : a list each tag_idx  
    Midddle : 
        sort_positions :  an array of start and end position[[ps1, pe1],...]
        sort_scores : an array of mean scores for each sort_positions's position pair
    Output : 
        n_best_phrases : a list of keyphrase str (number <= n_best)
        n_best_scores : scores for n_best_phrases
    '''
    # convert idx list to tag list
    pred_tags = converter.convert_idx2tag(indexs)
    
    # convert tag to positions array
    chunk_postions, chunk_scores = prepro_utils.get_chunk_positions(pred_tags, scores)
    
    # sort positions by mean score
    sort_positions = chunk_postions[np.argsort(-chunk_scores)] 
    sort_scores = chunk_scores[np.argsort(-chunk_scores)]
    
    n_best_phrases, n_best_scores = decode_pos2phrase(orig_tokens, sort_positions, sort_scores, n_best)
    return n_best_phrases, n_best_scores


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# soft decode phrases

def decode_ngram(orig_tokens, token_logits, converter, n, pooling=None):
    '''
    Combine n-gram score and sorted 
    Inputs :
        n : n_gram
        orig_tokens : document lower cased words' list 
        token_logits : each token has five score : for 'O', 'B', 'I', 'E', 'U' tag
        pooling : pooling method :  mean / min / log_mean
    Outputs : sorted phrase and socre list
    '''
    if n == 1:
        ngram_ids= converter.convert_tag2idx(['U'])
    elif n >= 2:
        ngram_ids = converter.convert_tag2idx(['B'] + ['I' for _ in range(n-2)] + ['E'])
    else:
        logger.info('invalid %d-gram !' %n)
    offsets = [i for i in range(len(ngram_ids))]
        
    # combine n-gram scores
    phrase_set = {}
    valid_length = (len(orig_tokens) - n + 1)
    for i in range(valid_length):
        n_gram = ' '.join(orig_tokens[i:i+n])
        
        if pooling == 'mean':
            n_gram_score = float(np.mean([token_logits[i+bias][tag] for bias, tag in zip(offsets, ngram_ids)]))
        elif pooling == 'min':
            n_gram_score = min([token_logits[i+bias][tag] for bias, tag in zip(offsets, ngram_ids)])
        elif pooling == 'log_mean':
            n_gram_score = float(np.mean([np.log(token_logits[i+bias][tag]) for bias, tag in zip(offsets, ngram_ids)]))
        else:
            logger.info('not %s pooling method !' % pooling)
            
        if n_gram not in phrase_set:
            phrase_set[n_gram] = n_gram_score
        else:
            phrase_set[n_gram] += n_gram_score
        
    phrase_list = []
    for phrase, score in phrase_set.items():
        phrase_list.append((phrase, score))
        
    sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
    return sorted_phrase_list
        


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return ' '.join([lower(x) for x in s]).rstrip()


def remove_empty(a_list):
    new_list = []
    for i in a_list:
        if len(i) > 0:
            if len(i[0]) >0:
                new_list.append(normalize_answer(i))   
    return new_list


def get_scores(candidate_KP, reference_KP):
    '''get em & f1 score '''
    em = getScoreEM(candidate_KP, reference_KP)  
    
    recall =  get_recall_score(candidate_KP, reference_KP) 
    precision = get_precision_score(candidate_KP, reference_KP)
    if precision == 0 or recall == 0:
        f1= 0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))
    return em, f1


def getScoreEM(candidate, reference):
    scoring, best_match = {}, {}
    max_score, max_label = 0 , ''
    for reference_label in reference:
        reference_key = str(reference_label)
        scoring[reference_key] = {}
        for candidate_label in candidate:
            candidate_key = str(candidate_label)
            if reference_label == candidate_label:
                scoring[reference_key][candidate_key] = 1
            else:
                scoring[reference_key][candidate_key] = 0
    while len(scoring) > 0:
        max_score = -1
        max_label = ''
        for reference_label in scoring:
            reference_key = str(reference_label)
            for candidate_label in scoring[reference_key]:
                candidate_key = str(candidate_label)
                score = scoring[reference_key][candidate_key]
                if score >= max_score:
                    max_score = score
                    max_label = (reference_key, candidate_key)
        best_match[max_label] = scoring[max_label[0]][max_label[1]]
        scoring.pop(max_label[0])
    return sum(best_match.values())/len(reference) 



def get_recall_score(candidate, reference):
    scoring, best_match = {}, {}
    max_score, max_label =  0,''
    set_candidate, set_reference = [], []
    for candidate_label in candidate:
        set_candidate.append(set(candidate_label))
    for reference_label in reference:
        set_reference.append(set(reference_label))
    for reference_label in set_reference:
        reference_key = str(reference_label)
        scoring[reference_key] = {}
        for candidate_label in set_candidate:
            candidate_key = str(candidate_label)
            scoring[reference_key][candidate_key] = (len(reference_label) - len(reference_label-candidate_label))/len(candidate_label)
    while len(scoring) > 0:
        max_score = 0
        max_label = ''
        for reference_label in scoring:
            reference_key = str(reference_label)
            for candidate_label in scoring[reference_key]:
                candidate_key = str(candidate_label)
                score = scoring[reference_key][candidate_key]
                if score >= max_score:
                    max_score = score
                    max_label = (reference_key, candidate_key)
        best_match[max_label] = scoring[max_label[0]][max_label[1]]
        scoring.pop(max_label[0])
    return sum(best_match.values())/len(reference)


def get_precision_score(candidate, reference):
    scoring, best_match = {}, {}
    max_score, max_label =  0,''
    set_candidate, set_reference = [], []
    for candidate_label in candidate:
        set_candidate.append(set(candidate_label))
    for reference_label in reference:
        set_reference.append(set(reference_label))
    for reference_label in set_reference:
        reference_key = str(reference_label)
        scoring[reference_key] = {}
        for candidate_label in set_candidate:
            candidate_key = str(candidate_label)
            scoring[reference_key][candidate_key] = (len(reference_label) - len(reference_label-candidate_label))/len(reference_label)
    while len(scoring) > 0:
        max_score = 0
        max_label = ''
        for reference_label in scoring:
            reference_key = str(reference_label)
            for candidate_label in scoring[reference_key]:
                candidate_key = str(candidate_label)
                score = scoring[reference_key][candidate_key]
                if score >= max_score:
                    max_score = score
                    max_label = (reference_key, candidate_key)
        best_match[max_label] = scoring[max_label[0]][max_label[1]]
        scoring.pop(max_label[0])
    return sum(best_match.values())/len(reference)
