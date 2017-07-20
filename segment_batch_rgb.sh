IMAGE_DIR_PATH='./datasets/test'
MODEL_PATH='./model/pspnet_rgb.pb'
IMAGE_EXTENSION='jpg'
OUTPUT_DIR='./datasets/outputs'

CUDA_VISIBLE_DEVICES=1 python test_batch_segmentation.py \
	--image_dir_path=${IMAGE_DIR_PATH} \
	--dataset_type='rgb' \
  	--model_path=${MODEL_PATH} \
  	--image_extension=${IMAGE_EXTENSION} \
  	--output_path=${OUTPUT_DIR} \
