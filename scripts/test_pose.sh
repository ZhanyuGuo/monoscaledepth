# This script is to eval pose quantity on seqence 9 & 10 in kitti odom dataset.

export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=~/checkpoint/pretrained_KITTI_MR/
MODEL_PATH=~/checkpoint/kitti_odom_20_multi_sup_new_1/models/weights_19/

DATA_PATH=~/dataset/KITTI_ODOM/dataset/

# EVAL_SPLIT=odom_9
EVAL_SPLIT=odom_10

python -m monoscaledepth.evaluate_pose \
    --load_weights_folder $MODEL_PATH \
    --data_path $DATA_PATH \
    --eval_split $EVAL_SPLIT \
