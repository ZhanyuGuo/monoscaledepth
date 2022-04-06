export CUDA_VISIBLE_DEVICES=1

DATA_PATH=~/dataset/kitti_raw_pose/dataset
LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_40_multi_sup
DATASET=kitti_raw_pose
SPLIT=kitti_raw_pose

EPOCHS=40
STEP_SIZE=30
FREEZE_EPOCHS=30
SAVE_FREQUENCY=1
BATCH_SIZE=8

SUP_EPOCHS=20
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
