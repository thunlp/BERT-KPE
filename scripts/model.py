import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from utils import override_args
from bertkpe import Idx2Tag, networks, generator, config_class
from bertkpe.transformers import AdamW, WarmupLinearSchedule
logger = logging.getLogger()


class KeyphraseSpanExtraction(object):
    
    def __init__(self, args, state_dict=None):
        
        self.args = args
        self.updates = 0
        
        # select model
        network = networks.get_class(args)
        
        # select config
        args.num_labels = 2 if args.model_class != 'bert2tag' else len(Idx2Tag)
        logger.info('Config num_labels = %d' %args.num_labels)
        model_config = config_class[args.pretrain_model_type].from_pretrained(args.cache_dir, num_labels=args.num_labels)
        
        # load pretrained model
        self.network = network.from_pretrained(args.cache_dir, config=model_config)
        
        # load checkpoint
        if state_dict is not None:
            self.network.load_state_dict(state_dict)
            logger.info('loaded checkpoint state_dict')
            
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def init_optimizer(self, num_total_steps):
        
        num_warmup_steps = int(self.args.warmup_proportion * num_total_steps)
        logger.info('warmup steps : %d' % num_warmup_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        param_optimizer = list(self.network.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=False)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)
        
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # train
    def update(self, step, inputs):
        # Train mode
        self.network.train()
        
        # run !
        loss = self.network(**inputs)

        if self.args.n_gpu > 1:
            # mean() to average on multi-gpu parallel (not distributed) training
            loss = loss.mean()
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step() 
            
            self.optimizer.zero_grad()
            self.updates += 1
        return loss.item()
    
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # test
    
    # bert2span
    def test_bert2span(self, inputs, lengths):
        self.network.eval()
        with torch.no_grad():
            s_logits, e_logits = self.network(**inputs)
            
        assert s_logits.size(0) == e_logits.size(0) == sum(lengths)
        s_logits = s_logits.data.cpu().tolist()
        e_logits = e_logits.data.cpu()

        start_lists, end_lists = [], []
        sum_len = 0
        for l in lengths:
            start_lists.append(s_logits[sum_len:(sum_len+l)])
            end_lists.append(e_logits[sum_len:(sum_len+l), :l].tolist())
            sum_len += l
        return start_lists, end_lists
    
    
    # bert2tag
    def test_bert2tag(self, inputs, lengths):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(**inputs)
            logits = F.softmax(logits, dim=-1)
        logits = logits.data.cpu().tolist()
        assert len(logits) == sum(lengths)
            
        logit_lists = []
        sum_len = 0
        for l in lengths:
            logit_lists.append(logits[sum_len:sum_len+l])
            sum_len += l
        return logit_lists
    
    # bert2chunk
    def test_bert2chunk(self, inputs, lengths, max_phrase_words):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(**inputs)
            logits = F.softmax(logits, dim=-1)
        logits = logits.data.cpu()[:,1]
        logits = logits.tolist()
        
        logit_lists = []
        sum_len = 0
        for l in lengths:
            batch_logit = []
            for n in range(max_phrase_words):
                batch_logit.append(logits[sum_len:sum_len+l-n])
                sum_len += (l-n)
            logit_lists.append(batch_logit)
        return logit_lists


    # bert2rank & bert2joint
    def test_bert2rank(self, inputs, numbers):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(**inputs) # shape = (batch_size, max_diff_gram_num)
            
        assert (logits.shape[0] == len(numbers)) and (logits.shape[1] == max(numbers))
        logits = logits.data.cpu().tolist()

        logit_lists = []
        for batch_id, num in enumerate(numbers):
            logit_lists.append(logits[batch_id][:num])
        return logit_lists
    
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, filename, epoch):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        params = {
            'args': self.args,
            'epoch': epoch,
            'state_dict': network.state_dict(),
        }
        try:
            torch.save(params, filename)
            logger.info('success save epoch_%d checkpoints !' % epoch)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
            
            
    @staticmethod
    def load_checkpoint(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(filename, map_location=lambda storage, loc:storage)
        
        args = saved_params['args']
        epoch = saved_params['epoch']
        state_dict = saved_params['state_dict']
        if new_args:
            args = override_args(args, new_args)
            
        model = KeyphraseSpanExtraction(args, state_dict)
        logger.info('success loaded epoch_%d checkpoints ! From : %s' % (epoch, filename))
        return model, epoch

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad() 
        # self.network.zero_grad()
        
    def set_device(self):
        self.network.to(self.args.device)
  
    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
        
    def distribute(self):
        self.distributed = True        
        self.network = torch.nn.parallel.DistributedDataParallel(self.network, 
                                                                 device_ids=[self.args.local_rank], 
                                                                 output_device=self.args.local_rank, 
                                                                 find_unused_parameters=True)
        
        