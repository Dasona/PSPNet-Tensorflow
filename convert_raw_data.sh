DATASET_DIR='./datasets'

CUDA_VISIBLE_DEVICES=1 python download_and_convert_data.py \
	--dataset_name='ade20k' \
	--dataset_dir=${OUTPUT_DIR} \
	--dataset_type='rgb' \
