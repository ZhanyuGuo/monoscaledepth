TARGET_IMG_PATH=assets/0000000019.jpg
SOURCE_IMG_PATH=assets/0000000018.jpg
DEPTH_PATH=assets/0000000019_multi_disp_resized.npy
INTRINSICS=assets/test_sequence_intrinsics.json
SAVE_PATH=visualization/
# CROP_BEYOND=100

python -m visualization.pointcloud_in_camera \
    --image_path $TARGET_IMG_PATH \
    --depth_path $DEPTH_PATH \
    --intrinsics_path $INTRINSICS \
    --save_path $SAVE_PATH \
    # --crop_beyond $CROP_BEYOND \

# sudo apt install meshlab
# meshlab scene.obj
