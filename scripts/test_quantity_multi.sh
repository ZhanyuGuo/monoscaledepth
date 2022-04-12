export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=~/checkpoint/kitti_raw_20_12_multi/models/weights_19/
DATA_PATH=~/dataset/KITTI_RAW

python -m monoscaledepth.evaluate_depth \
   --load_weights_folder $MODEL_PATH \
   --data_path $DATA_PATH \
   --eval_mono \
   # --disable_median_scaling \
