# This script is to test quality using mono depth.

export CUDA_VISIBLE_DEVICES=0

TARGET_IMG_PATH=assets/0000000019.jpg
MODEL_PATH=~/checkpoint/kitti_raw_20_mono_sup_new/models/weights_19/

python -m monoscaledepth.test_simple \
    --target_image_path $TARGET_IMG_PATH \
    --model_path $MODEL_PATH \
    --no_multi_depth \
