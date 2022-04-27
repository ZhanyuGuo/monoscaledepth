# This script is to generate pointcloud in camera coordinates by 
# using the target image, depth estimation and camera intrinsics. 
# Saved as `.obj` file and can be opened by meshlab, etc.

# Q: How to open .obj?
# A: $ sudo apt install meshlab
#    $ meshlab scene.obj

# NOTE You should run `test_quality_*.sh` first to generate a depth file.

TARGET_IMG_PATH=assets/0000000019.jpg

# DEPTH_PATH=assets/0000000019_multi_disp_resized.npy
DEPTH_PATH=assets/0000000019_disp_resized.npy

INTRINSICS=assets/test_sequence_intrinsics.json
SAVE_PATH=assets/
# CROP_BEYOND=100

python -m visualization.pointcloud_in_camera \
    --image_path $TARGET_IMG_PATH \
    --depth_path $DEPTH_PATH \
    --intrinsics_path $INTRINSICS \
    --save_path $SAVE_PATH \
    # --crop_beyond $CROP_BEYOND \
