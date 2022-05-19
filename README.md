# README

## Environment

- Change your anaconda's channels to get the best download speed.

- Configure the environment.

  ```bash
  conda env create -f environment.yml
  ```

- Activate the environment.

  ```bash
  conda activate monoscaledepth
  ```

## Dataset

- KITTI Raw Dataset, CITY category, [link](http://www.cvlibs.net/datasets/kitti/raw_data.php).
- KITTI Odometry Dataset, [link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Custom Dataset, e.g. our Dominant Dataset, which was captured in the underground garage.
- Make3D. (only for generalization experiment) (TODO)

## Prepare Data

- If you use KITTI Dataset above, just unzip them anywhere but keep their file structure.
- If you use Custom Dataset.
  - (optional) Transform from video to images with specified size, i.e. 640x192. (data/video_to_image.py & data/copy_rename_data.py)
  - Concatenate k(e.g. 3) continuous images into one image. (data/prepare_train_data.py or scripts/prepare_dominant.sh)
  - Design a dataloader to load concatenated image.
  - Train.

## Train

- Use command line.

  ``` bash
  python -m monoscaledepth.train
  ```

- Use scripts.

  ```bash
  bash scripts/train_mono.sh
  bash scripts/train_mono_sup.sh
  bash scripts/train_multi.sh
  bash scripts/train_multi_sup.sh
  # and many other ablation.
  ```

## Evaluation

- Qualitative.

  ```bash
  bash scripts/test_quality_full.sh
  bash scripts/test_quality_mono.sh
  bash scripts/test_quality_multi.sh
  ```

- Quantitative.

  ```bash
  bash scripts/test_quantity_mono.sh
  bash scripts/test_quantity_multi.sh
  ```

(TODO)