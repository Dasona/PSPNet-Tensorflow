#!/bin/bash

PATH_TO_FREEZED_MODEL="/home/aman/PSPNet-Tensorflow/model/pspnet_rgb.pb"

CUDA_VISIBLE_DEVICES=1 python demo/demo.py \
	--debug \
	--gpu \
	--model=${PATH_TO_FREEZED_MODEL} \
