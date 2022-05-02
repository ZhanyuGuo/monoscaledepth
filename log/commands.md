# Commands

### 连接服务器

```bash
# Old
ssh -p 99 gzy@192.168.111.137
# New
ssh -p 36045 -i ~/gzy_rsa gzy@ae61b2716f3e6691.natapp.cc

source ~/.bashrc
conda activate manydepth
```

```bash
# tensorboard
ssh -L 16006:127.0.0.1:6006 -p 36045 -i ~/gzy_rsa gzy@ae61b2716f3e6691.natapp.cc
tensorboard --logdir <log_dir>
```

### 两帧图片生成深度图

```bash
python -m manydepth.test_simple \
    --target_image_path assets/test_sequence_target.jpg \
    --source_image_path assets/test_sequence_source.jpg \
    --intrinsics_json_path assets/test_sequence_intrinsics.json \
    --model_path <MODEL_PATH>
```

### 生成与查看点云

```bash
python visualization/pointcloud_camera_coor.py \
    --image_path visualization/10902.jpg \
    --depth_path visualization/10902_20.npy \
    --intrinsics_path assets/dominant_intrinsics.json \
    --save_path visualization/ \
    --crop_beyond 0
```

```bash
meshlab visualization/scene.obj
```

### 训练步骤

```bash
python data/prepare_train_data.py \
   --dataset_dir ~/dataset/dominant/ \
   --dataset_name dominant \
   --dump_root ~/dataset/dominant/<FOLDER_NAME>/ \
   --seq_length 3 \
   --img_height 192 \
   --img_width 640 \
   --sample_gap 2
```

```bash
python data/prepare_train_data.py \
   --dataset_dir ~/dataset/KITTI_ODOM/dataset/ \
   --dataset_name kitti_odom_pose \
   --dump_root ~/dataset/kitti_odom_pose/dataset/ \
   --seq_length 3 \
   --img_height 192 \
   --img_width 640
```

```bash
python data/prepare_train_data.py \
   --dataset_dir ~/dataset/KITTI_ODOM/dataset/ \
   --dataset_name kitti_odom_pose \
   --dump_root ~/dataset/kitti_odom_pose/dataset_randGap2/ \
   --seq_length 3 \
   --img_height 192 \
   --img_width 640 
   # --sample_gap 2
```

```bash
python data/prepare_train_data.py \
   --dataset_name kitti_raw_pose_eigen \
   --dataset_dir ~/dataset/KITTI_RAW/ \
   --dump_root ~/dataset/kitti_raw_pose/dataset_test/ \
   --seq_length 3 \
   --img_height 192 \
   --img_width 640 
```

然后将 train_files.txt 和 val_files.txt 放入 splits/dominant 中, 开始训练.

```bash
python -m manydepth.train \
   --data_path ~/dataset/dominant/split_3/ \
   --log_dir ~/checkpoint/ \
   --model_name dominant \
   --dataset dominant \
   --split dominant \
   --batch_size 10 \
   --num_epochs 100 \
   --scheduler_step_size 75 \
   --freeze_teacher_epoch 75 \
   --save_frequency 10
```

>maybe batch size of 10 for 12GB GPU for `640x192` (the "MR" resolution).
>
>[How much memory does this cost?](https://github.com/nianticlabs/manydepth/issues/16)

```bash
python -m manydepth.train \
   --data_path ~/dataset/dominant/split_5/ \
   --log_dir ~/checkpoint/ \
   --model_name dominant \
   --dataset dominant \
   --split dominant \
   --batch_size 6 \
   --frame_ids 0 -1 1 -2 2 \
   --num_matching_frames 2
```

```bash
python -m manydepth.train \
   --data_path ~/dataset/dominant/split_3/ \
   --log_dir ~/checkpoint/ \
   --model_name dominant \
   --dataset dominant \
   --split dominant \
   --batch_size 10 \
   --depth_binning inverse
```

### git强制pull远程仓库覆盖本地代码

```bash
git fetch --all
git reset --hard origin/master
```

## Screen常用指令

```bash
screen -S train
screen -r train
```

删除环境, 先进入环境后输入

```bash
exit
```
