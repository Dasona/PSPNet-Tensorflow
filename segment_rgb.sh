#!/bin/bash

IMAGE_PATH='abc.jpg'
DATASET_DIR='datasets/records'
CHECKPOINT_PATH='./train/imagenet_best'

CUDA_VISIBLE_DEVICES="" python test_segmentation.py \
	--dataset_dir=${DATASET_DIR} \
	--dataset_type='rgb' \
	--model_name='pspnet_rgb' \
	--checkpoint_path=${CHECKPOINT_PATH} \
	--eval_image_size=473 \
  	--num_classes=150 \
  	--image=${IMAGE_PATH} \
