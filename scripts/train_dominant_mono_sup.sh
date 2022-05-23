# This script is to train the model with multi depth, with pose supervised.

export CUDA_VISIBLE_DEVICES=1
DATA_PATH=~/dataset/dominant/new/

LOG_PATH=~/checkpoint/
MODEL_NAME=dominant_mono_sup_50_1
DATASET=dominant_pose

SPLIT=dominant

EPOCHS=50
STEP_SIZE=35
SAVE_FREQUENCY=10
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
   --save_frequency $SAVE_FREQUENCY \
   --add_pose_supervise \
   --begin_supervise_epoch $SUP_EPOCHS \
   --pose_weight $POSE_WEIGHT \
   --no_multi_depth \
