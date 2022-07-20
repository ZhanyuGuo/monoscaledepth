# This script is to train the model with multi depth, without pose supervised.

export CUDA_VISIBLE_DEVICES=1

DATA_PATH=~/dataset/KITTI_RAW/
LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_multi_20
DATASET=kitti_raw
SPLIT=eigen_zhou

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=8
SEED=612

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
   --pytorch_random_seed $SEED \
