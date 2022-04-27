# This script is to train the model without multi depth and without pose supervised.

export CUDA_VISIBLE_DEVICES=1

# DATA_PATH=~/dataset/kitti_raw_pose/dataset
DATA_PATH=~/dataset/KITTI_RAW/

LOG_PATH=~/checkpoint/

MODEL_NAME=kitti_raw_20_mono_3
DATASET=kitti_raw_pose

# SPLIT=kitti_raw_pose
SPLIT=eigen_zhou

EPOCHS=20
STEP_SIZE=15
SAVE_FREQUENCY=1
BATCH_SIZE=12

python -m monoscaledepth.train \
   --data_path $DATA_PATH \
   --log_dir $LOG_PATH \
   --model_name $MODEL_NAME \
   --dataset $DATASET \
   --split $SPLIT \
   --batch_size $BATCH_SIZE \
   --num_epochs $EPOCHS \
   --scheduler_step_size $STEP_SIZE \
   --save_frequency $SAVE_FREQUENCY \
   --no_multi_depth \
