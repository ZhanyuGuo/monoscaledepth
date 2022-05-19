# This script is to train the model with multi depth, with pose supervised.

export CUDA_VISIBLE_DEVICES=1

# DATA_PATH=~/dataset/kitti_raw_pose/dataset/
DATA_PATH=~/dataset/KITTI_RAW/

LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_20_0_multi_sup_new_7_frames_1
DATASET=kitti_raw_pose

# SPLIT=kitti_raw_pose
SPLIT=eigen_zhou

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=4
FRAME_IDS="0 -1 1 -2 2 -3 3"
NUM_MATCHING_FRAMES=3

SUP_EPOCHS=0
POSE_WEIGHT=0.05

python -m monoscaledepth.train \
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
   --frame_ids $FRAME_IDS \
   --num_matching_frames $NUM_MATCHING_FRAMES \
