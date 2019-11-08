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

logger = logging.getLogger()
torch.backends.cudnn.benchmark=True 

from Constants import Idx2Tag
from prepro_utils import IdxTag_Converter
from model import KeyphraseSpanExtraction
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, data_loader, model, stats, writer):
    logger.info("start training %d epoch ..." % stats['epoch'])
    
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    
    epoch_loss = 0
    epoch_step = 0
    
    for step, batch in enumerate(tqdm(data_loader)):
        
        try:
            loss = model.update(step, batch)
        except:
            logging.error(str(traceback.format_exc()))
            continue
            
        train_loss.update(loss)
        
        epoch_loss += loss
        epoch_step += 1
                
        if step % args.display_iter == 0:
            
            if args.local_rank in [-1, 0] and args.use_viso:
                writer.add_scalar('train/loss', train_loss.avg, model.updates)
                writer.add_scalar('train/lr', model.scheduler.get_lr()[0], model.updates)
                
            logging.info('train: Epoch = %d | iter = %d/%d | ' %
                        (stats['epoch'], step, len(train_data_loader)) +
                        'loss = %.2f | %d updates | elapsed time = %.2f (s) \n' %
                        (train_loss.avg, model.updates, stats['timer'].time()))
            train_loss.reset()            
            
    logging.info('Epoch Mean Loss = %.4f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 ((epoch_loss / epoch_step), stats['epoch'], epoch_time.time()))
    
    
def evaluate(args, data_loader, model, stats, writer):
    logger.info("start evaluate valid ( %d epoch ) ..." % stats['epoch'])

    epoch_time = utils.Timer()
    
    epoch_loss = 0
    epoch_step = 0
    
    for step, batch in enumerate(tqdm(data_loader)):
        try:
            loss = model.predict(batch)
        except:
            logging.error(str(traceback.format_exc()))
            continue
            
        epoch_loss += loss
        epoch_step += 1
       
    eval_loss = float((epoch_loss / epoch_step))
    
    if args.local_rank in [-1, 0] and args.use_viso:
        writer.add_scalar('valid/loss', eval_loss, stats['epoch'])
        
    logging.info('Valid Evaluation | Epoch Mean Loss = %.4f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 (eval_loss, stats['epoch'], epoch_time.time()))
    
    return eval_loss

# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser('BERT-Seq-Tagging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    # init tokenizer & Converter 
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)
    converter = IdxTag_Converter(Idx2Tag)

    # -------------------------------------------------------------------------------------------
    # build dataloaders
    logger.info("start loading openkp datasets ...")
    dataset_dict = utils.read_openkp_examples(args, tokenizer)
    
    # train dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = utils.build_openkp_dataset(args, dataset_dict['train'], tokenizer, converter)
    
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_train_eval,
        pin_memory=args.cuda,
    )
    
    # valid dataset
    args.valid_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    valid_dataset = utils.build_openkp_dataset(args, dataset_dict['valid'], tokenizer, converter) # don't use DistributedSampler
    
    valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        sampler=valid_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_train_eval,
        shuffle=False,
        pin_memory=args.cuda,
    )
    
    # -------------------------------------------------------------------------------------------
    # Set training total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.max_train_epochs = args.max_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.max_train_epochs
        
   # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    logger.info(" ************************** Initilize Model & Optimizer ************************** ")
    
    if args.load_checkpoint and os.path.isfile(args.checkpoint_file):
        model = KeyphraseSpanExtraction.load_checkpoint(args.checkpoint_file, args)        
    else:
        logger.info('Training model from scratch...')
        model = KeyphraseSpanExtraction(args)
    model.init_optimizer(num_total_steps=t_total)
    
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    model.set_device()
    if args.n_gpu > 1:
        model.parallelize()
    
    if args.local_rank != -1:
        model.distribute()
        
    if args.local_rank == 0:
        torch.distributed.barrier()
        
    if args.local_rank in [-1, 0] and args.use_viso:
        tb_writer = SummaryWriter(args.viso_folder)
    else:
        tb_writer = None
        
    logger.info("Training/evaluation parameters %s", args)
    logger.info(" ************************** Running training ************************** ")
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", args.max_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info(" *********************************************************************** ")
    
    # -------------------------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------------------------
    model.zero_grad()
    stats = {'timer': utils.Timer(), 'epoch': 0, 'min_eval_loss': float("inf")}
    
    for epoch in range(1, (args.max_train_epochs+1)):
        stats['epoch'] = epoch
        
        # train 
        train(args, train_data_loader, model, stats, tb_writer)
        
        # eval & test
        eval_loss = evaluate(args, valid_data_loader, model, stats, tb_writer)
        if eval_loss < stats['min_eval_loss']:
            stats['min_eval_loss'] = eval_loss
            logger.info(" *********************************************************************** ")
            logger.info('Update Min Eval_Loss = %.6f (epoch = %d)'% (stats['min_eval_loss'], stats['epoch']))
            
        # Checkpoint
        if args.save_checkpoint:
            model.save_checkpoint(os.path.join(args.output_folder, 'epoch_{}.checkpoint'.format(epoch)), stats['epoch'])