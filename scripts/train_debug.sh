# This script is to train the model with multi depth and with pose supervised.

export CUDA_VISIBLE_DEVICES=0

DATA_PATH=~/dataset/KITTI_RAW/
LOG_PATH=~/checkpoint/
MODEL_NAME=debug
DATASET=kitti_raw_pose
SPLIT=eigen_zhou

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=3
SEED=612

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
   --pytorch_random_seed $SEED \
   --add_pose_supervise \
   --begin_supervise_epoch $SUP_EPOCHS \
   --pose_weight $POSE_WEIGHT \



# python3 -m debugpy --listen 5678 --wait-for-client -m monoscaledepth.train \
#    --data_path $DATA_PATH \
#    --log_dir $LOG_PATH \
#    --model_name $MODEL_NAME \
#    --dataset $DATASET \
#    --split $SPLIT \
#    --batch_size $BATCH_SIZE \
#    --num_epochs $EPOCHS \
#    --scheduler_step_size $STEP_SIZE \
#    --freeze_teacher_epoch $FREEZE_EPOCHS \
#    --save_frequency $SAVE_FREQUENCY \
#    --add_pose_supervise \
#    --begin_supervise_epoch $SUP_EPOCHS \
#    --pose_weight $POSE_WEIGHT \
#    --use_semantic \
