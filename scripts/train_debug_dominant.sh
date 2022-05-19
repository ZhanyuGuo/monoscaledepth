# This script is to train the model with multi depth and with pose supervised.

export CUDA_VISIBLE_DEVICES=0

# DATA_PATH=~/dataset/kitti_raw_pose/dataset/
# DATA_PATH=~/dataset/KITTI_RAW/
DATA_PATH=~/dataset/dominant/new/

LOG_PATH=~/checkpoint/

# MODEL_NAME=kitti_raw_20_multi_sup_semantic_debug
MODEL_NAME=dominant_sup_100_debug

# DATASET=kitti_raw_pose_semantic
DATASET=dominant_pose

# SPLIT=eigen_zhou
SPLIT=dominant

EPOCHS=100
STEP_SIZE=75
FREEZE_EPOCHS=75
SAVE_FREQUENCY=10
BATCH_SIZE=3

SUP_EPOCHS=0
POSE_WEIGHT=0.05

# python3 -m pip install debugpy
# Copy launch.json into .vscode

python3 -m debugpy --listen 5678 --wait-for-client -m monoscaledepth.train \
   --data_path $DATA_PATH \
   --log_dir $LOG_PATH \
   --model_name $MODEL_NAME \
   --dataset $DATASET \
   --split $SPLIT \
   --batch_size $BATCH_SIZE \
   --num_epochs $EPOCHS \
   --scheduler_step_size $STEP_SIZE \
   --freeze_teacher_epoch $FREEZE_EPOCHS \
   --save_frequency $SAVE_FREQUENCY \
   --add_pose_supervise \
   --begin_supervise_epoch $SUP_EPOCHS \
   --pose_weight $POSE_WEIGHT \
