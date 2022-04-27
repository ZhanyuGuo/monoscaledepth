# This script is to train the model with multi depth and with pose supervised loading pretrained model.

export CUDA_VISIBLE_DEVICES=1

# DATA_PATH=~/dataset/kitti_raw_pose/dataset/
DATA_PATH=~/dataset/KITTI_RAW/

LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_10_multi_pretrained_3
DATASET=kitti_raw_pose

# SPLIT=kitti_raw_pose
SPLIT=eigen_zhou

EPOCHS=10
STEP_SIZE=5
FREEZE_EPOCHS=3
SAVE_FREQUENCY=1
BATCH_SIZE=8

SUP_EPOCHS=0
POSE_WEIGHT=0.05

WEIGHT_FOLDER=~/checkpoint/important_kitti_raw_20/models/weights_19

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
   --load_weights_folder $WEIGHT_FOLDER \
   --mono_weights_folder $WEIGHT_FOLDER \
