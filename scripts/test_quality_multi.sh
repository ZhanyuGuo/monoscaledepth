export CUDA_VISIBLE_DEVICES=0

TARGET_IMG_PATH=assets/19_000387.png
SOURCE_IMG_PATH=assets/19_000386.png
INTRINSICS=assets/test_sequence_intrinsics.json

MODEL_PATH=~/checkpoint/kitti_raw_20_multi/models/weights_19/

python -m monoscaledepth.test_simple \
    --target_image_path $TARGET_IMG_PATH \
    --source_image_path $SOURCE_IMG_PATH \
    --intrinsics_json_path $INTRINSICS \
    --model_path $MODEL_PATH \
