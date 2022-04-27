# This script is to train the model with multi depth and without pose supervised on kitti odom.

export CUDA_VISIBLE_DEVICES=1

DATA_PATH=~/dataset/KITTI_ODOM/dataset

LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_odom_20_multi_new
DATASET=kitti_odom_pose

SPLIT=odom

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=10

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
