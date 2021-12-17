## *************************************
## English
## *************************************
export DATA_PATH=../data

export dataset_class=openkp # openkp , kp20k
export max_train_steps=20810 #  20810 (openkp) , 73430 (kp20k)

export model_class=bert2joint # bert2span, bert2tag, bert2chunk, bert2rank, bert2joint
export pretrain_model=roberta-base # bert-base-cased , spanbert-base-cased , roberta-base

## --------------------------------------------------------------------------------
## DataParallel (Multi-GPUs)

CUDA_VISIBLE_DEVICES=0,1 python train.py --run_mode train \
--local_rank -1 \
--max_train_steps $max_train_steps \
--model_class $model_class \
--dataset_class $dataset_class \
--pretrain_model_type $pretrain_model \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--per_gpu_test_batch_size 32 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--display_iter 200 \
--save_checkpoint \
--use_viso \


# ## --------------------------------------------------------------------------------
# ## Distributed-DataParallel (Multi-GPUs)
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --run_mode train \
# --max_train_steps $max_train_steps \
# --model_class $model_class \
# --dataset_class $dataset_class \
# --pretrain_model_type $pretrain_model \
# --per_gpu_train_batch_size 8 \
# --gradient_accumulation_steps 4 \
# --per_gpu_test_batch_size 32 \
# --preprocess_folder $DATA_PATH/prepro_dataset \
# --pretrain_model_path $DATA_PATH/pretrain_model \
# --cached_features_dir $DATA_PATH/cached_features \
# --display_iter 200 \
# --save_checkpoint \
# --use_viso \



## *************************************
## *************************************
## Multilingual
## *************************************
## *************************************

# export DATA_PATH=../data

# export dataset_class=multidata # openkp , kp20k, multidata
# export max_train_steps=5000 #  20810 (openkp) , 73430 (kp20k)

# export model_class=bert2joint # bert2span, bert2tag, bert2chunk, bert2rank, bert2joint
# export pretrain_model=distilbert-base-multilingual-cased # bert-base-cased , spanbert-base-cased , roberta-base

## --------------------------------------------------------------------------------
## DataParallel (Multi-GPUs)

# CUDA_VISIBLE_DEVICES=0 python3 train.py --run_mode train \
# --local_rank -1 \
# --max_train_steps $max_train_steps \
# --max_train_epochs 10 \
# --model_class $model_class \
# --dataset_class $dataset_class \
# --pretrain_model_type $pretrain_model \
# --per_gpu_train_batch_size 16 \
# --gradient_accumulation_steps 4 \
# --per_gpu_test_batch_size 32 \
# --preprocess_folder $DATA_PATH/prepro_dataset \
# --pretrain_model_path $DATA_PATH/pretrain_model \
# --cached_features_dir $DATA_PATH/cached_features \
# --display_iter 200 \
# --save_checkpoint \
# --use_viso \



# ## --------------------------------------------------------------------------------
# ## Distributed-DataParallel (Multi-GPUs)
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --run_mode train \
# --max_train_steps $max_train_steps \
# --model_class $model_class \
# --dataset_class $dataset_class \
# --pretrain_model_type $pretrain_model \
# --per_gpu_train_batch_size 8 \
# --gradient_accumulation_steps 4 \
# --per_gpu_test_batch_size 32 \
# --preprocess_folder $DATA_PATH/prepro_dataset \
# --pretrain_model_path $DATA_PATH/pretrain_model \
# --cached_features_dir $DATA_PATH/cached_features \
# --display_iter 200 \
# --save_checkpoint \
# --use_viso \
