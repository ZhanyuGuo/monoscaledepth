# This script is to test quality fully by using target image,
# source image, camera intrinsics. Visualize the result.

export CUDA_VISIBLE_DEVICES=0

# TARGET_IMG_PATH=assets/0000000191.jpg
# SOURCE_IMG_PATH=assets/0000000190.jpg

# TARGET_IMG_PATH=assets/0000000167.jpg
# SOURCE_IMG_PATH=assets/0000000166.jpg

# TARGET_IMG_PATH=assets/0000000019.jpg
# SOURCE_IMG_PATH=assets/0000000018.jpg

# TARGET_IMG_PATH=assets/test_quality_1_target.jpg
# SOURCE_IMG_PATH=assets/test_quality_1_source.jpg

# TARGET_IMG_PATH=assets/test_quality_2_target.jpg
# SOURCE_IMG_PATH=assets/test_quality_2_source.jpg

# TARGET_IMG_PATH=assets/test_quality_3_target.jpg
# SOURCE_IMG_PATH=assets/test_quality_3_source.jpg

TARGET_IMG_PATH=assets/test_quality_4_target.jpg
SOURCE_IMG_PATH=assets/test_quality_4_source.jpg

INTRINSICS=assets/test_sequence_intrinsics.json
MODEL_PATH=~/checkpoint/kitti_raw_20_0_multi_sup_new_bin/models/weights_19/

python -m monoscaledepth.test_full \
    --target_image_path $TARGET_IMG_PATH \
    --source_image_path $SOURCE_IMG_PATH \
    --intrinsics_json_path $INTRINSICS \
    --model_path $MODEL_PATH \
