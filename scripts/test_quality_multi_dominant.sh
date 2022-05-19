# This script is to test quality using multi depth.

export CUDA_VISIBLE_DEVICES=1

TARGET_IMG_PATH=assets/10563.jpg
SOURCE_IMG_PATH=assets/10562.jpg
INTRINSICS=assets/dominant_intrinsics.json

MODEL_PATH=~/checkpoint/dominant_sup_20/models/weights_19/

python -m monoscaledepth.test_simple \
    --target_image_path $TARGET_IMG_PATH \
    --source_image_path $SOURCE_IMG_PATH \
    --intrinsics_json_path $INTRINSICS \
    --model_path $MODEL_PATH \
