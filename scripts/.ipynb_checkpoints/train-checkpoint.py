import os
import sys
import time
import tqdm
import json
import torch
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import test
import utils
import config
from model import KeyphraseSpanExtraction
from utils import pred_arranger, pred_saver
from bertkpe import dataloader, generator, evaluator
from bertkpe import tokenizer_class, Idx2Tag, Tag2Idx, Decode_Candidate_Number
# from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = True
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

logger = logging.getLogger()

# Trainer
# -------------------------------------------------------------------------------------------
def train(args, data_loader, model, train_input_refactor, stats, writer):
    logger.info(
        "start training %s on %s (%d epoch) || local_rank = %d..."
        % (args.model_class, args.dataset_class, stats["epoch"], args.local_rank)
    )

    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 0

    scaler = torch.cuda.amp.GradScaler()

    epoch_iterator = tqdm(
        data_loader, desc="Train_Iteration", disable=args.local_rank not in [-1, 0]
    )
    for step, batch in enumerate(epoch_iterator):
        inputs, indices = train_input_refactor(batch, model.args.device)
        try:
            loss = model.update(step, inputs, scaler)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        train_loss.update(loss)
        epoch_loss += loss
        epoch_step += 1

        if args.local_rank in [-1, 0] and step % args.display_iter == 0:
            if args.use_viso:
                writer.add_scalar("train/loss", train_loss.avg, model.updates)
                writer.add_scalar(
                    "train/lr", model.scheduler.get_lr()[0], model.updates
                )

            logging.info(
                "Local Rank = %d | train: Epoch = %d | iter = %d/%d | "
                % (args.local_rank, stats["epoch"], step, len(train_data_loader))
                + "loss = %.4f | lr = %f | %d updates | elapsed time = %.2f (s) \n"
                % (
                    train_loss.avg,
                    model.scheduler.get_lr()[0],
                    model.updates,
                    stats["timer"].time(),
                )
            )
            train_loss.reset()

    logging.info(
        "Local Rank = %d | Epoch Mean Loss = %.8f ( Epoch = %d ) | Time for epoch = %.2f (s) \n"
        % (
            args.local_rank,
            (epoch_loss / epoch_step),
            stats["epoch"],
            epoch_time.time(),
        )
    )


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # setting args
    parser = argparse.ArgumentParser(
        "BertKPE", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config.add_default_args(parser)
    args = parser.parse_args()
    config.init_args_config(args)
    # -------------------------------------------------------------------------------------------
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # -------------------------------------------------------------------------------------------
    utils.set_seed(args)
    # Make sure only the first process in distributed training will download model & vocab

    # -------------------------------------------------------------------------------------------
    # init tokenizer & Converter
    logger.info(
        "start setting tokenizer, dataset and dataloader (local_rank = {})... ".format(
            args.local_rank
        )
    )
    tokenizer = tokenizer_class[args.pretrain_model_type].from_pretrained(
        args.cache_dir
    )

    # -------------------------------------------------------------------------------------------
    # Select dataloader
    batchify_features_for_train, batchify_features_for_test = dataloader.get_class(
        args.model_class
    )

    # -------------------------------------------------------------------------------------------
    # build train dataloader
    train_dataset = dataloader.build_dataset(
        **{"args": args, "tokenizer": tokenizer, "mode": "train"}
    )
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        torch.utils.data.sampler.RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify_features_for_train,
        pin_memory=args.cuda,
    )
    logger.info("Successfully Preprocess Training Features !")

    # -------------------------------------------------------------------------------------------
    # build dev dataloader
    dev_dataset = dataloader.build_dataset(
        **{"args": args, "tokenizer": tokenizer, "mode": "dev"}
    )
    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify_features_for_test,
        pin_memory=args.cuda,
    )
    logger.info("Successfully Preprocess Dev Features !")

    # -------------------------------------------------------------------------------------------
    # build eval dataloader
    if args.dataset_class == "kp20k":
        eval_dataset = dataloader.build_dataset(
            **{"args": args, "tokenizer": tokenizer, "mode": "eval"}
        )
        eval_sampler = torch.utils.data.sampler.SequentialSampler(eval_dataset)
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.test_batch_size,
            sampler=eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )
        logger.info("Successfully Preprocess Eval Features !")

    # -------------------------------------------------------------------------------------------
    # Set training total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.max_train_epochs = (
            args.max_train_steps
            // (len(train_data_loader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_data_loader)
            // args.gradient_accumulation_steps
            * args.max_train_epochs
        )

    # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info(
        " ************************** Initilize Model & Optimizer ************************** "
    )

    if args.load_checkpoint and os.path.isfile(args.checkpoint_file):
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(
            args.checkpoint_file, args
        )
    else:
        logger.info("Training model from scratch...")
        model = KeyphraseSpanExtraction(args)

    # initial optimizer
    model.init_optimizer(num_total_steps=t_total)

    # -------------------------------------------------------------------------------------------
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # -------------------------------------------------------------------------------------------

    # set model device
    model.set_device()

    if args.n_gpu > 1:
        model.parallelize()

    if args.local_rank != -1:
        model.distribute()

    if args.local_rank in [-1, 0] and args.use_viso:
        tb_writer = SummaryWriter(args.viso_folder)
    else:
        tb_writer = None

    logger.info("Training/evaluation parameters %s", args)
    logger.info(
        " ************************** Running training ************************** "
    )
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", args.max_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info(
        " *********************************************************************** "
    )

    # -------------------------------------------------------------------------------------------
    # Method Select
    candidate_decoder = test.select_decoder(args.model_class)
    evaluate_script, main_metric_name = utils.select_eval_script(args.dataset_class)
    train_input_refactor, test_input_refactor = utils.select_input_refactor(
        args.model_class
    )

    # -------------------------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------------------------
    model.zero_grad()
    stats = {"timer": utils.Timer(), "epoch": 0, main_metric_name: 0}
    for epoch in range(1, (args.max_train_epochs + 1)):
        stats["epoch"] = epoch

        # train
        train(args, train_data_loader, model, train_input_refactor, stats, tb_writer)

        # previous metric score
        prev_metric_score = stats[main_metric_name]

        # decode candidate phrases
        dev_candidate = candidate_decoder(
            args,
            dev_data_loader,
            dev_dataset,
            model,
            test_input_refactor,
            pred_arranger,
            "dev",
        )
        stats = evaluate_script(
            args, dev_candidate, stats, mode="dev", metric_name=main_metric_name
        )

        # new metric score
        new_metric_score = stats[main_metric_name]

        # save checkpoint : when new metric score > previous metric score
        if (
            args.save_checkpoint
            and (new_metric_score > prev_metric_score)
            and (args.local_rank == -1 or torch.distributed.get_rank() == 0)
        ):
            checkpoint_name = "{}.{}.{}.epoch_{}.checkpoint".format(
                args.model_class,
                args.dataset_class,
                args.pretrain_model_type.split("-")[0],
                epoch,
            )
            model.save_checkpoint(
                os.path.join(args.checkpoint_folder, checkpoint_name), stats["epoch"]
            )

        # eval evaluation
        if args.dataset_class == "kp20k":
            eval_candidate = candidate_decoder(
                args,
                eval_data_loader,
                eval_dataset,
                model,
                test_input_refactor,
                pred_arranger,
                "eval",
            )
            eval_stats = {"epoch": epoch, main_metric_name: 0}
            eval_stats = evaluate_script(
                args,
                eval_candidate,
                eval_stats,
                mode="eval",
                metric_name=main_metric_name,
            )
