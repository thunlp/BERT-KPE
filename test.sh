CUDA_VISIBLE_DEVICES=2,3 python test.py --run_mode generate \
--local_rank -1 \
--pooling min \
--eval_checkpoint ./DATA/pretrain_model/output/epoch_4.checkpoint \
--log_dir ./Log \
--pred_dir ./Pred \