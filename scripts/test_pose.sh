export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=~/checkpoint/kitti_raw_20_multi_sup/models/weights_19/
# MODEL_PATH=~/checkpoint/pretrained_KITTI_MR/

DATA_PATH=~/dataset/KITTI_ODOM/dataset/

python -m monoscaledepth.evaluate_pose \
    --eval_split odom_9 \
    --load_weights_folder $MODEL_PATH \
    --data_path $DATA_PATH \
