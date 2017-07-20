#!/bin/bash

EVAL_DIR='./eval/pspnet'
DATASET_DIR='datasets/records'
CHECKPOINT_PATH='./train/psp-new'

CUDA_VISIBLE_DEVICES="" python eval_semantic_segmentation.py \
	--eval_dir=${EVAL_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name='ade20k' \
	--dataset_type='rgb' \
	--model_name='pspnet_rgb' \
	--dataset_split_name='validation' \
	--checkpoint_path=${CHECKPOINT_PATH} \
	--eval_image_size=473 \
	--batch_size=2 \
	--preprocessing_name='rgb' \
	--eval_image_size=473 \
	--crop_larger_dim=512 \
  	--crop_smaller_dim=473 \
  	--num_classes=37 \
  	--training_size=9000 \
  	--validation_size=1000


