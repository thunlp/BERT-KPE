CUDA_VISIBLE_DEVICES=2,3  python train.py --run_mode train \
--local_rank -1 \
--max_train_epochs 5 \
--save_checkpoint \
--log_dir ./Log \
--output_dir ./output \

# --use_viso \
