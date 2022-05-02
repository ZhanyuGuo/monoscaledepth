# This script is to train the model with multi depth, with pose supervised, and with semantic masks.

export CUDA_VISIBLE_DEVICES=1

DATA_PATH=~/dataset/KITTI_RAW/

LOG_PATH=~/checkpoint/
MODEL_NAME=kitti_raw_20_0_multi_sup_semantic_m2_pre_1
DATASET=kitti_raw_pose_semantic

SPLIT=eigen_zhou

EPOCHS=20
STEP_SIZE=15
FREEZE_EPOCHS=15
SAVE_FREQUENCY=1
BATCH_SIZE=8

SUP_EPOCHS=0
POSE_WEIGHT=0.05

WEIGHT_FOLDER=~/checkpoint/kitti_raw_20_0_multi_sup_new/models/weights_19

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
   --use_semantic \
   --load_weights_folder $WEIGHT_FOLDER \
   --mono_weights_folder $WEIGHT_FOLDER \
