# This script is to test quality using multi depth.

export CUDA_VISIBLE_DEVICES=0

# TARGET_IMG_PATH=assets/10563.jpg

# TARGET_IMG_PATH=assets/10764.jpg

TARGET_IMG_PATH=assets/10902.jpg

MODEL_PATH=~/checkpoint/dominant_mono_sup_50_0/models/weights_49/

python -m monoscaledepth.test_simple \
    --target_image_path $TARGET_IMG_PATH \
    --model_path $MODEL_PATH \
    --no_multi_depth \

