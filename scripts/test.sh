export DATA_PATH=../data

CUDA_VISIBLE_DEVICES=0 python test.py --run_mode test \
--model_class bert2joint \
--local_rank -1 \
--pretrain_model_type distilbert-base-multilingual-cased \
--dataset_class openkp \
--per_gpu_test_batch_size 64 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--eval_checkpoint /home/amitchaulwar/PoC/BERT-KPE/results/train_bert2joint_openkp_distilbert_11.01_14.45/checkpoints/bert2joint.openkp.distilbert.epoch_11.checkpoint \