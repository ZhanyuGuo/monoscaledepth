# This script is to train the model with multi depth, with pose supervised.

export CUDA_VISIBLE_DEVICES=1
DATA_PATH=~/dataset/dominant/new/

LOG_PATH=~/checkpoint/
MODEL_NAME=dominant_sup_20_1
DATASET=dominant_pose

SPLIT=dominant

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=10

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
