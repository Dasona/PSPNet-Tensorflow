#!/bin/bash

TRAIN_DIR='train/psp-new/'
DATASET_DIR='datasets/records'
CHECKPOINT_PATH='./train/imagenet_best'
CHECKPOINT_EXCLUDE_SCOPES='global_step:0,'

CUDA_VISIBLE_DEVICES=0 python train_semantic_segmentation.py \
  --train_dir=${TRAIN_DIR} \
  --num_readers=4 \
  --num_preprocessing_threads=4 \
  --log_every_n_steps=10 \
  --save_summaries_secs=60 \
  --save_interval_secs=60 \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name='ade20k' \
  --dataset_split_name='training' \
  --model_name='pspnet_rgb' \
  --preprocessing_name='rgb' \
  --dataset_type='rgb' \
  --optimizer='momentum' \
  --weight_decay=0.0001 \
  --learning_rate=0.001 \
  --end_learning_rate=0.00001 \
  --learning_rate_decay_type='polynomial' \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_delay = 2.0 \
  --train_image_size=473 \
  --batch_size=4 \
  --max_number_of_steps=100000 \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE_SCOPES} \
  --ignore_missing_vars=True \
  --crop_larger_dim=512 \
  --crop_smaller_dim=473 \
  --num_classes=150 \
  --training_size=20210 \
  --validation_size=2000
