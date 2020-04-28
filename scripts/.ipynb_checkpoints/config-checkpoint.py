import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()


def add_default_args(parser):
    # ---------------------------------------------------------------------------------------------
    # mode select
    modes = parser.add_argument_group('Modes')
    modes.add_argument('--run_mode', type=str, choices=['train', 'test'],
                       help='Select running mode. ')
    modes.add_argument('--dataset_class', type=str, choices=['openkp', 'kp20k'],
                       help='Select datasets.')
    modes.add_argument('--model_class', type=str, 
                       choices=['bert2span', 'bert2tag', 'bert2chunk', 'bert2rank', 'bert2joint'],
                       help='Select different model types.')
    modes.add_argument("--pretrain_model_type", type=str,
                       choices=['bert-base-cased', 'spanbert-base-cased', 'roberta-base'],
                       help="Select pretrain model type.")
    # ---------------------------------------------------------------------------------------------
    # Filesystem
    files = parser.add_argument_group('Files')
    files.add_argument('--preprocess_folder', type=str, required=True,
                       help='Directory of preprocess data.')
    files.add_argument("--pretrain_model_path", type=str, required=True,
                       help="Path to pre-trained BERT model.")
    files.add_argument("--save_path", type=str, default='../results',
                       help="Path to save log files, checkpoint and prediction results.")
    files.add_argument("--cached_features_dir", type=str, default=None,
                       help="Filepath used to reload preprocessed data features.")
    # ---------------------------------------------------------------------------------------------
    # Runtime environment
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--no_cuda', action='store_true', default=False,
                         help='Train Model on GPUs (False)')
    runtime.add_argument("--local_rank", type=int, default=0,
                        help="Set local_rank=0 for distributed training on multiple gpus")
    runtime.add_argument('--data_workers', type=int, default=0,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # ---------------------------------------------------------------------------------------------
    # Train parameters
    train = parser.add_argument_group('Training')
    train.add_argument("--max_token", type=int, default=510, 
                       help="Max length of document WordPiece tokens + '[CLS]'+'[SEP]' <= 512.")
    train.add_argument("--max_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform. ")
    train.add_argument("--max_train_steps", default=0, type=int,
                        help="Total number of training steps. ")
    train.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    train.add_argument("--per_gpu_test_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for test, orignal = 128")
    train.add_argument("--gradient_accumulation_steps", default=4, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # ---------------------------------------------------------------------------------------------
    # Optimizer
    optim = parser.add_argument_group('Optimizer')
    optim.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    optim.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    optim.add_argument("--warmup_proportion", default=0.1, type=float,
                       help="Linear warmup over warmup_ratio warm_step / t_total.")
    optim.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    optim.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # ---------------------------------------------------------------------------------------------
    # Evaluation
    evaluate = parser.add_argument_group('Evaluation')
    evaluate.add_argument("--tag_pooling", default='min', type=str, 
                          help="Pooling methods for Bert2Tag.")
    evaluate.add_argument("--eval_checkpoint", default='', type=str, 
                          help="Tha checkpoint file to be evaluated. ")
    evaluate.add_argument("--max_phrase_words", default=5, type=int,
                        help="The max length of phrases. ")
    
    # ---------------------------------------------------------------------------------------------
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--use_viso', action='store_true', default=False,
                         help='Whether use tensorboadX to log loss.')
    general.add_argument('--display_iter', type=int, default=200,
                         help='Log state after every <display_iter> batches.')
    general.add_argument('--load_checkpoint', action='store_true', default=False,
                        help='Path to a checkpoint for generation .')
    general.add_argument("--checkpoint_file", default='', type=str, 
                         help="Load checkpoint model continue training.")
    general.add_argument('--save_checkpoint', action='store_true', default=False,
                        help='If true, Save model + optimizer state after each epoch.')
    general.add_argument('--server_ip', type=str, default='', 
                         help="For distant debugging.")
    general.add_argument('--server_port', type=str, default='', 
                         help="For distant debugging.")
    general.add_argument('--fp16', action='store_true', default=False,
                         help="Whether to use 16-bit float precision instead of 32-bit")
    general.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    
    # bert pretrained model
    args.cache_dir = os.path.join(args.pretrain_model_path, args.pretrain_model_type)
    args.preprocess_folder = os.path.join(args.preprocess_folder, args.dataset_class)
    
    # Result folder
    folder_name = "%s_%s_%s_%s_" %(args.run_mode, args.model_class, args.dataset_class, 
                                  args.pretrain_model_type.split('-')[0]) + time.strftime("%m.%d_%H.%M")
    args.save_folder = os.path.join(args.save_path, folder_name)
    
    # principle folder
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    # viso folder
    if args.use_viso:
        args.viso_folder = os.path.join(args.save_folder, 'viso')
        if not os.path.exists(args.viso_folder):
            os.mkdir(args.viso_folder)
            
    # checkpoint folder
    if args.run_mode == 'train':
        args.checkpoint_folder = os.path.join(args.save_folder, 'checkpoints')
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
            
    elif args.run_mode == 'test':
        args.pred_folder = os.path.join(args.save_folder, 'predictions')
        if not os.path.exists(args.pred_folder):
            os.mkdir(args.pred_folder)
    
    # logging file
    args.log_file = os.path.join(args.save_folder, 'logging.txt')
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
    logger.info("preprocess_folder = {}".format(args.preprocess_folder))
    logger.info("Pretrain Model Type = {}".format(args.pretrain_model_type))