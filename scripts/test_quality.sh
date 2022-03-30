export CUDA_VISIBLE_DEVICES=0

IMG_PATH=assets/19_000387.png
MODEL_PATH=~/checkpoint/kitti_raw_20_mono_sup/models/weights_19/

python -m monoscaledepth.test_simple \
    --image_path $IMG_PATH \
    --model_path $MODEL_PATH \
