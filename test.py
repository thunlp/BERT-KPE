import os
import time
import tqdm
import json
import torch
import random
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

import utils
import config
import eval_utils
from Constants import Idx2Tag
from prepro_utils import IdxTag_Converter
from model import KeyphraseSpanExtraction
from transformers import BertTokenizer
from eval_utils import remove_empty, normalize_answer
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

logger = logging.getLogger()
torch.backends.cudnn.benchmark=True



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
# -------------------------------------------------------------------------------------------
# get prediction keyphrases
# -------------------------------------------------------------------------------------------
def decode_n_best_candidates(orig_tokens, token_logits, converter, max_gram, topk):
    '''
    max_gram :  type :int , max_phrase_words
    topk : type int , consider top_k phrase to evaluate metrics .
    return : phrase token list & score list
    '''
    assert len(orig_tokens) == len(token_logits)
    orig_tokens = [token.lower() for token in orig_tokens]

    ngrams = []
    for n in range(1, max_gram+1):
        ngrams.extend(eval_utils.decode_ngram(orig_tokens, token_logits, converter, n, args.pooling))
    # sorted all n-grams
    sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)
    
    n_best_phrases = [phrase.split() for phrase, _ in sorted_ngrams[:topk]]
    n_best_scores = [score for _, score in sorted_ngrams[:topk]]
    return n_best_phrases, n_best_scores


# -------------------------------------------------------------------------------------------
# generate candidate for Public_Eval
# -------------------------------------------------------------------------------------------
def generate_candidates(dataset, data_loader, model, mode):
    logging.info(' Start Generating Keyphrases for %s  ...'% mode)
    test_time = utils.Timer()     

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(tqdm(data_loader)):
        try:
            logit_lists = model.test(batch)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        example_indices, example_lengths = batch[-1], batch[-2]
        for batch_id, logit_list in enumerate(logit_lists):
            
            example = dataset.examples[example_indices[batch_id]]
            
            # example info
            valid_length = example_lengths[batch_id]
            cut_tokens = example['orig_tokens'][:valid_length]

            # decode tag to phrase and sort
            n_best_phrases, n_best_scores = decode_n_best_candidates(cut_tokens, logit_list, 
                                                                     converter, args.max_phrase_words, 100)
            candidate_KP = remove_empty(n_best_phrases)[:5]
            assert len(candidate_KP) == 5
                
            # log groundtruths & predictions
            tot_predictions.append((example['url'], candidate_KP))
            tot_examples += 1
    logging.info('Dataset: %s Finish Generation | Total Examples = %d | predict time = %.2f (s)' 
                 % (mode, tot_examples, test_time.time()))
    return tot_predictions



def save_prediction(args, tot_predictions, mode):
    filename = os.path.join(args.pred_folder, '%s_candidate.json'% mode)
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for prediction in tot_predictions:
            data = {}
            data['url'] = prediction[0]
            data['KeyPhrases'] = [keyphrase.split() for keyphrase in prediction[1]]
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save %s prediction to %s' %(mode, filename))

    
# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser('BERT-SeqTagging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)
    
    args = parser.parse_args()

    # initilize agrs & config
    config.init_args_config(args)
    args.num_labels = len(Idx2Tag)

    # -------------------------------------------------------------------------------------------
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # -------------------------------------------------------------------------------------------
    set_seed(args)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    # -------------------------------------------------------------------------------------------
    # Load Model & Optimizer
    # -------------------------------------------------------------------------------------------
    logger.info(" ************************** Initilize Model & Optimizer ************************** ")
    try:
        model = KeyphraseSpanExtraction.load_checkpoint(args.eval_checkpoint, args)
        model.set_device()

    except ValueError:
        print("Could't Load Pretrain Model %s" % args.eval_checkpoint)
        
    if args.n_gpu > 1:
        model.parallelize()
    
    if args.local_rank != -1:
        model.distribute()
        
    if args.local_rank == 0:
        torch.distributed.barrier()

    # -------------------------------------------------------------------------------------------
    # init tokenizer & Converter 
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)
    converter = IdxTag_Converter(Idx2Tag)
    
    # -------------------------------------------------------------------------------------------
    # build dataloaders
    logger.info("start loading openkp datasets ...")
    dataset_dict = utils.read_openkp_examples(args, tokenizer)
    
    # -----------------------------------------------------------------------------------------------------------
    # Dev dataloader
    args.eval_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    
    dev_dataset = utils.build_openkp_dataset(args, dataset_dict['valid'], tokenizer, converter)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_test,
        shuffle=False,
        pin_memory=args.cuda,
    )

    # -----------------------------------------------------------------------------------------------------------
    # Eval dataloader
    eval_dataset = utils.build_openkp_dataset(args, dataset_dict['eval_public'], tokenizer, converter)
    eval_sampler = torch.utils.data.sampler.SequentialSampler(eval_dataset)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_test,
        shuffle=False,
        pin_memory=args.cuda,
    )

    # generate candidates 
    dev_tot_predictions = generate_candidates(dev_dataset, dev_data_loader, model, 'Dev')
    eval_tot_predictions = generate_candidates(eval_dataset, eval_data_loader, model, 'EvalPublic')

    # save candidates
    save_prediction(args, dev_tot_predictions, 'Dev')
    save_prediction(args, eval_tot_predictions, 'EvalPublic')

