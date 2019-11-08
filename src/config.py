import os
import sys
import time
import logging
import argparse

MODEL_DIR = "./DATA/pretrain_model/"
preprocess_folder = "./DATA/cached_features"



def add_default_args(parser):
    
    # ---------------------------------------------------------------------------------------------
    # Filesystem
    files = parser.add_argument_group('Files')
    
    files.add_argument('--run_mode', type=str, default='train',
                       help='select a mode for train model or generate prediction from checkpoint')
    files.add_argument('--preprocess_folder', type=str, default=preprocess_folder,
                       help='Directory of preprocess data')
    files.add_argument("--pretrain_model_path", type=str, default=MODEL_DIR,
                       help="Path to pre-trained model.")
    files.add_argument("--cache_folder", type=str, default='bert-base-cased',
                       help="shortcut name for BERT model ")
    files.add_argument('--log_dir', type=str, default='./Log',
                       help='saved logfiles')
    files.add_argument('--pred_dir', type=str, default='./Pred',
                       help='saved evaluations results and generations')
    files.add_argument('--output_dir', type=str, default='./output', help='path to save model')
    
    
    # ---------------------------------------------------------------------------------------------
    # Runtime environment
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--no_cuda', action='store_true', default=False,
                         help='Train Model on GPUs (False)')
    runtime.add_argument("--local_rank", type=int, default=-1,
                    help="set ocal_rank=0 for distributed training on multiple gpus")
    runtime.add_argument('--data_workers', type=int, default=2,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    runtime.add_argument('--split_seed', type=int, default=5,
                        help="random seed for split valid and test dataset .")
    runtime.add_argument('--fp16', action='store_true', default=False,
                         help="Whether to use 16-bit float precision instead of 32-bit")
    runtime.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # ---------------------------------------------------------------------------------------------
    # train parameters
    train = parser.add_argument_group('Training')
    train.add_argument("--max_src_len", type=int, default=510, 
                       help="the max length of document WordPiece tokens + '[CLS]'+'[SEP]' <= 512.")
    train.add_argument("--max_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform. ")
    train.add_argument("--max_train_steps", default=0, type=int,
                        help="Total number of training steps. ")
    train.add_argument("--per_gpu_train_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")
    train.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    train.add_argument("--per_gpu_test_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for test.")
    train.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # ---------------------------------------------------------------------------------------------
    # Optimizer
    optim = parser.add_argument_group('Optimizer')
    
    optim.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    optim.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some. 0.01?")
    optim.add_argument("--warmup_proportion", default=0, type=int,
                       help="Linear warmup over warmup_ratio warm_step / t_total.")
    optim.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    optim.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # ---------------------------------------------------------------------------------------------
    # eval 
    evaluate = parser.add_argument_group('Evaluation')
    
    evaluate.add_argument("--pooling", default='min', type=str, 
                          help="pooling methods : min / mean / log_mean.")
    evaluate.add_argument("--eval_checkpoint", default='', type=str, 
                          help="use to save .checkpoint file")
    evaluate.add_argument("--max_phrase_words", default=5, type=int,
                        help=": eval and test dataset all = 6")
    
    # ---------------------------------------------------------------------------------------------
    # General
    general = parser.add_argument_group('General')

    general.add_argument('--use_viso', action='store_true', default=False,
                         help='whether use tensorboadX to log loss')
    general.add_argument('--display_iter', type=int, default=100,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--load_checkpoint', action='store_true', default=False,
                        help='the path to a checkpoint for generation .')
    general.add_argument("--checkpoint_file", default='', type=str, 
                         help="loaded checkpoint model continue training .")
    general.add_argument('--save_checkpoint', action='store_true', default=False,
                        help='If true, Save model + optimizer state after each epoch .')
    general.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    general.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    
def init_args_config(args):
    
    args.cache_dir = os.path.join(args.pretrain_model_path, args.cache_folder)
    args.model_name = '%s-' % args.run_mode + time.strftime("%Y.%m.%d-%H:%M:%S")
    args.log_folder = os.path.join(args.log_dir, args.model_name)

    # log folder
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)
    args.log_file = os.path.join(args.log_folder, 'logging.txt')
        
    # model folder
    if args.run_mode == 'train':
        args.output_folder = os.path.join(args.output_dir, args.model_name)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
            
        # viso folder
        if args.use_viso:
            args.viso_folder = os.path.join(args.log_folder, 'viso')
            if not os.path.exists(args.viso_folder):
                os.mkdir(args.viso_folder)

    # eval folder            
    elif args.run_mode == 'generate':
        args.pred_folder = os.path.join(args.pred_dir, args.model_name)
        if not os.path.exists(args.pred_dir):
            os.mkdir(args.pred_dir)
        if not os.path.exists(args.pred_folder):
            os.mkdir(args.pred_folder)
            
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    
    

