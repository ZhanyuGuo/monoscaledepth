# This script is to test quality fully by using target image,
# source image, camera intrinsics. Visualize the result.

export CUDA_VISIBLE_DEVICES=0

INPUT_FOLDER=~/dataset/KITTI_RAW/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/
OUTPUT_FOLDER=~/output_images/
INTRINSICS=assets/test_sequence_intrinsics.json
MODEL_PATH=~/checkpoint/kitti_raw_20_0_multi_sup_semantic_m2_pre_1/models/weights_19/

python -m monoscaledepth.test_full_folder \
    --input_folder $INPUT_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --intrinsics_json_path $INTRINSICS \
    --model_path $MODEL_PATH \
    # --no_cuda \
