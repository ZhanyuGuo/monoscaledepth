# This script is to prepare dominant data.

DATA_DIR=~/dataset/dominant/
DATASET=dominant
DUMP_ROOT=~/dataset/dominant/new/
SEQ_LENGTH=3
IMG_HEIGHT=192
IMG_WIDTH=640
SAMPLE_GAP=1

python data/prepare_train_data.py \
   --dataset_dir $DATA_DIR \
   --dataset_name $DATASET \
   --dump_root $DUMP_ROOT \
   --seq_length $SEQ_LENGTH \
   --img_height $IMG_HEIGHT \
   --img_width $IMG_WIDTH \
   --sample_gap $SAMPLE_GAP \
