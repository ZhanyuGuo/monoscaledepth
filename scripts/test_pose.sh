export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=~/checkpoint/kitti_raw_20_multi_sup/models/weights_19/
# MODEL_PATH=~/checkpoint/pretrained_KITTI_MR/

DATA_PATH=~/dataset/KITTI_ODOM/dataset/

# EVAL_SPLIT=odom_9
EVAL_SPLIT=odom_10

python -m monoscaledepth.evaluate_pose \
    --eval_split $EVAL_SPLIT \
    --load_weights_folder $MODEL_PATH \
    --data_path $DATA_PATH \
