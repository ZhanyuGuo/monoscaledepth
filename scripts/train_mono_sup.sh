export CUDA_VISIBLE_DEVICES=1

DATA_PATH=~/dataset/kitti_raw_pose/dataset
LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_20_5_mono_sup
DATASET=kitti_raw_pose
SPLIT=kitti_raw_pose

EPOCHS=20
STEP_SIZE=15
SAVE_FREQUENCY=1
BATCH_SIZE=12

SUP_EPOCHS=5
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
   --no_multi_depth \
   --add_pose_supervise \
   --begin_supervise_epoch $SUP_EPOCHS \
   --pose_weight $POSE_WEIGHT \
