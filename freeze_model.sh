CHECKPOINT_PATH='./train/imagenet_best'
OUTPUT_DIR='./model'
OUTPUT_FILE_NAME='pspnet_rgb.pb'

CUDA_VISIBLE_DEVICES=1 python save_model.py \
	--checkpoint_path=${CHECKPOINT_PATH} \
	--output_dir=${OUTPUT_DIR} \
	--model_name='pspnet_rgb' \
	--output_filename=${OUTPUT_FILE_NAME} \
	--num_classes=150 \

