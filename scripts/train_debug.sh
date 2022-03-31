export CUDA_VISIBLE_DEVICES=0

DATA_PATH=~/dataset/kitti_raw_pose/dataset/
LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_20_mono_sup_debug
DATASET=kitti_raw_pose
SPLIT=kitti_raw_pose

EPOCHS=20
STEP_SIZE=15
SAVE_FREQUENCY=1
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
   --save_frequency $SAVE_FREQUENCY \
   --no_multi_depth \
   --add_pose_supervise \
   --begin_supervise_epoch $SUP_EPOCHS
   --pose_weight $POSE_WEIGHT \
