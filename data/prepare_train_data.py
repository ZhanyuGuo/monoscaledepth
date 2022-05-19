from __future__ import division

import os
import argparse
import scipy.misc
import numpy as np
import imageio

from glob import glob
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--dump_root", type=str, required=True)
parser.add_argument("--seq_length", type=int, default=3)
parser.add_argument("--img_height", type=int, default=192)
parser.add_argument("--img_width", type=int, default=640)
parser.add_argument("--sample_gap", type=int, default=1)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument(
    "--dataset_name",
    type=str,
    choices=[
        "kitti_raw_eigen",
        "kitti_raw_pose_eigen",
        "kitti_raw_stereo",
        "kitti_odom",
        "kitti_odom_pose",
        "cityscapes",
        "dominant",
    ],
    required=True,
)
args = parser.parse_args()


def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, args):
    if n % 20 == 0:
        print("Progress %d/%d...." % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example["image_seq"])
    intrinsics = example["intrinsics"]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    dump_dir = os.path.join(args.dump_root, example["folder_name"])

    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + "/%s.jpg" % example["file_name"]

    imageio.imwrite(dump_img_file, image_seq.astype(np.uint8))
    # scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))

    dump_cam_file = dump_dir + "/%s_cam.txt" % example["file_name"]
    with open(dump_cam_file, "w") as f:
        f.write("%f,0.,%f,0.,%f,%f,0.,0.,1." % (fx, cx, fy, cy))

    if args.dataset_name in ["dominant", "kitti_odom_pose", "kitti_raw_pose_eigen"]:
        pose = example["gt_pose"]
        dump_pose_file = dump_dir + "/%s_pose.txt" % example["file_name"]
        with open(dump_pose_file, "w") as f:
            f.write(pose)


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == "kitti_odom":
        from kitti.kitti_odom_loader import kitti_odom_loader

        data_loader = kitti_odom_loader(
            args.dataset_dir,
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "kitti_odom_pose":
        from kitti.kitti_odom_pose_loader import kitti_odom_pose_loader

        data_loader = kitti_odom_pose_loader(
            args.dataset_dir,
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "kitti_raw_eigen":
        from kitti.kitti_raw_loader import kitti_raw_loader

        data_loader = kitti_raw_loader(
            args.dataset_dir,
            split="eigen",
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "kitti_raw_pose_eigen":
        from kitti.kitti_raw_pose_loader import kitti_raw_pose_loader

        data_loader = kitti_raw_pose_loader(
            args.dataset_dir,
            split="eigen",
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "kitti_raw_stereo":
        from kitti.kitti_raw_loader import kitti_raw_loader

        data_loader = kitti_raw_loader(
            args.dataset_dir,
            split="stereo",
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "cityscapes":
        from cityscapes.cityscapes_loader import cityscapes_loader

        data_loader = cityscapes_loader(
            args.dataset_dir,
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
        )

    if args.dataset_name == "dominant":
        from dominant.dominant_loader import dominant_loader

        data_loader = dominant_loader(
            args.dataset_dir,
            img_height=args.img_height,
            img_width=args.img_width,
            seq_length=args.seq_length,
            sample_gap=args.sample_gap,
        )

    Parallel(n_jobs=args.num_threads)(
        delayed(dump_example)(n, args) for n in range(data_loader.num_train)
    )

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + "train_files.txt", "w") as tf:
        with open(args.dump_root + "val_files.txt", "w") as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + "/%s" % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, "*.jpg"))
                frame_ids = [os.path.basename(fi).split(".")[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write("%s %s\n" % (s, frame))
                    else:
                        tf.write("%s %s\n" % (s, frame))


main()
