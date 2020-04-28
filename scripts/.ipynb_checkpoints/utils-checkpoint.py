import os
import time
import json
import torch
import random
import argparse
import logging
import numpy as np
from bertkpe import evaluate_openkp, evaluate_kp20k

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Select Input Refactor
# -------------------------------------------------------------------------------------------
def select_input_refactor(name):
    if name == 'bert2span':
        return train_input_refactor_bert2span, test_input_refactor_bert2span
    elif name in ['bert2tag', 'bert2chunk', 'bert2rank']:
        return train_input_refactor, test_input_refactor
    elif name == 'bert2joint':
        return train_input_refactor_bert2joint, test_input_refactor
    raise RuntimeError('Invalid retriever class: %s' % name)

    
# -------------------------------------------------------------------------------------------
# Bert2Span
def train_input_refactor_bert2span(batch, device):
    ex_indices = batch[-1]
    batch = tuple(b.to(device) for b in batch[:-1])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'valid_output':batch[3],
              'active_mask':batch[4],
              's_label':batch[5],
              'e_label':batch[6],
              'end_mask':batch[7]
             }        
    return inputs, ex_indices

def test_input_refactor_bert2span(batch, device):
    ex_indices, ex_lengths = batch[-1], batch[-2]
    batch = tuple(b.to(device) for b in batch[:-2])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'valid_output':batch[3],
              'active_mask':batch[4]
             }        
    return inputs, ex_indices, ex_lengths


# -------------------------------------------------------------------------------------------
# bert2tag, bert2chunk, bert2rank
def train_input_refactor(batch, device):
    ex_indices = batch[-1]
    batch = tuple(b.to(device) for b in batch[:-1])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'active_mask':batch[3],
              'valid_output':batch[4],
              'labels':batch[5]
             }        
    return inputs, ex_indices

def test_input_refactor(batch, device):
    ex_indices, ex_phrase_numbers = batch[-1], batch[-2]
    batch = tuple(b.to(device) for b in batch[:-2])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'active_mask':batch[3],
              'valid_output':batch[4],
             }
    return inputs, ex_indices, ex_phrase_numbers

# -------------------------------------------------------------------------------------------
# bert2joint
def train_input_refactor_bert2joint(batch, device):
    ex_indices = batch[-1]
    batch = tuple(b.to(device) for b in batch[:-1])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'active_mask':batch[3],
              'valid_output':batch[4],
              'labels':batch[5],
              'chunk_labels':batch[6],
              'chunk_mask':batch[7],
             }        
    return inputs, ex_indices

# -------------------------------------------------------------------------------------------
# Select Prediction Arranger
# -------------------------------------------------------------------------------------------
def pred_arranger(tot_predictions):
    data_dict = {}
    for prediction in tot_predictions:
        item = {}
        item['url'] = prediction[0]
        item['KeyPhrases'] = [keyphrase.split() for keyphrase in prediction[1]]
        if len(prediction) > 2:
            item['Scores'] = prediction[2]
        data_dict[item['url']] = item
    return data_dict


def pred_saver(args, tot_predictions, filename):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for url, item in tot_predictions.items():
            data = {}
            data['url'] = url
            data['KeyPhrases'] = item['KeyPhrases']
            if "Scores" in item:
                data['Scores'] = item['Scores']
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save %s prediction file' % filename)
    

# -------------------------------------------------------------------------------------------
# Select Evaluation Scripts
# -------------------------------------------------------------------------------------------

def select_eval_script(name):
    if name == 'openkp':
        return openkp_evaluate_script, "max_f1_score3"
    elif name == 'kp20k':
        return kp20k_evaluate_script, "max_f1_score5"
    raise RuntimeError('Invalid retriever class: %s' % name)
    

# OpenKP Evaluation Script
def openkp_evaluate_script(args, candidate, stats, mode, metric_name='max_f1_score3'):
    logger.info("*"*80)
    logger.info("Start Dev Evaluatng : Epoch = %d" % stats['epoch'])
    epoch_time = Timer()
    
    reference_filename = os.path.join(args.preprocess_folder, 'openkp.dev_candidate.json') 
    f1_scores, precision_scores, recall_scores = evaluate_openkp(candidate, reference_filename)

    for i in precision_scores:
        logger.info("@{}".format(i))
        logger.info("F1:{}".format(np.mean(f1_scores[i])))
        logger.info("P:{}".format(np.mean(precision_scores[i])))
        logger.info("R:{}".format(np.mean(recall_scores[i])))
        
    f1_score3 = np.mean(f1_scores[3])
    if f1_score3 > stats[metric_name]:
        logger.info("-"*60)
        stats[metric_name] = f1_score3
        logger.info('Update ! Update ! Update ! Max f1_score3 = %.4f (epoch = %d, local_rank = %d)' 
                    %(stats[metric_name], stats['epoch'], args.local_rank))
        logger.info("-"*60)
    logger.info("Local Rank = %d ||End Dev Evaluatng : Epoch = %d (Time: %.2f (s)) " 
                %(args.local_rank, stats['epoch'], epoch_time.time()))
    logger.info("*"*80)
    return stats


# KP20k Evaluation Script
def kp20k_evaluate_script(args, candidate, stats, mode, metric_name='max_f1_score5'):
    logger.info("*"*80)
    logger.info("Start Evaluatng : Mode = %s || Epoch = %d" % (mode, stats['epoch']))
    epoch_time = Timer()
    
    reference_filename = os.path.join(args.preprocess_folder, 'kp20k.%s_candidate.json' %mode) 
    f1_scores, precision_scores, recall_scores = evaluate_kp20k(candidate, reference_filename)

    for i in precision_scores:
        logger.info("@{}".format(i))
        logger.info("F1:{}".format(np.mean(f1_scores[i])))
        logger.info("P:{}".format(np.mean(precision_scores[i])))
        logger.info("R:{}".format(np.mean(recall_scores[i])))
        
    f1_score5 = np.mean(f1_scores[5])
    if f1_score5 > stats[metric_name]:
        logger.info("-"*60)
        stats[metric_name] = f1_score5
        logger.info('Update ! Update ! Update ! ||  Mode = %s || Max f1_score5 = %.4f (epoch = %d, local_rank = %d)' 
                    %(mode, stats[metric_name], stats['epoch'], args.local_rank))
        logger.info("-"*60)
    logger.info("Local Rank = %d || End Evaluatng : Mode = %s || Epoch = %d (Time: %.2f (s)) " 
                %(args.local_rank, mode, stats['epoch'], epoch_time.time()))
    logger.info("*"*80)
    return stats

    
# -------------------------------------------------------------------------------------------
# Common Functions
# -------------------------------------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def override_args(old_args, new_args):
    ''' cover old args to new args, log which args has been changed.'''
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k in old_args:
            if old_args[k] != new_args[k]:
                logger.info('Overriding saved %s: %s --> %s' %(k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
        else:
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
    
