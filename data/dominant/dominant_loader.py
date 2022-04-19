from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob
from random import random


class dominant_loader(object):
    def __init__(
        self,
        dataset_dir,
        split="train",
        sample_gap=1,
        img_height=192,
        img_width=640,
        seq_length=3,
    ):
        self.dataset_dir = dataset_dir
        self.split = split

        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, "seq_length must be odd!"
        self.frames = self.collect_frames(split)
        self.num_frames = len(self.frames)
        if split == "train":
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print("Total frames collected: %d" % self.num_frames)

    def collect_frames(self, split):
        img_dir = self.dataset_dir + "raw_data/" + split + "/"
        folders = os.listdir(img_dir)
        frames = []
        for folder in folders:
            img_files = glob(img_dir + folder + "/*.jpg")
            for f in img_files:
                frame_id = os.path.basename(f).split(".")[0]
                frames.append(frame_id)
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_frame_id):
            return False
        example = self.load_example(self.frames[tgt_idx])
        return example

    def load_intrinsics(self, frame_id, split):
        fx = 389.941
        fy = 390.065
        u0 = 314.603
        v0 = 96.151
        intrinsics = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
        return intrinsics

    def load_pose(self, frame_id, gap):
        folder = "garage"
        half_offset = int((self.seq_length - 1) / 2 * gap)
        pose_seq = []
        for o in range(-half_offset, half_offset + 1, gap):
            curr_frame_id = "%.5d" % (int(frame_id) + o)
            curr_pose_file = os.path.join(
                self.dataset_dir, "raw_data", self.split, folder, curr_frame_id + ".txt"
            )
            with open(curr_pose_file, "r") as f:
                curr_pose = f.readline() + "\n"
            pose_seq.append(curr_pose)

        pose = "".join(pose_seq)
        return pose

    def is_valid_example(self, tgt_frame_id):
        folder, tgt_local_frame_id = "garage", tgt_frame_id
        half_offset = int((self.seq_length - 1) / 2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = "%.5d" % (int(tgt_local_frame_id) + o)
            curr_frame_id = curr_local_frame_id
            curr_image_file = os.path.join(
                self.dataset_dir, "raw_data", self.split, folder, curr_frame_id + ".jpg"
            )
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, gap):
        folder, tgt_local_frame_id = "garage", tgt_frame_id
        half_offset = int((self.seq_length - 1) / 2 * gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, gap):
            curr_local_frame_id = "%.5d" % (int(tgt_local_frame_id) + o)
            curr_frame_id = curr_local_frame_id
            curr_image_file = os.path.join(
                self.dataset_dir, "raw_data", self.split, folder, curr_frame_id + ".jpg"
            )
            curr_img = scipy.misc.imread(curr_image_file)
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height / raw_shape[0]
                zoom_x = self.img_width / raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))

            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, tgt_frame_id):
        if self.sample_gap == 2:
            n = random()
            if n < 0.5:
                gap = 1
            else:
                gap = 2
        else:
            gap = self.sample_gap

        image_seq, zoom_x, zoom_y = self.load_image_sequence(
            tgt_frame_id, self.seq_length, gap
        )
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        pose = self.load_pose(tgt_frame_id, gap)
        
        example = {}
        example["intrinsics"] = intrinsics
        example["image_seq"] = image_seq
        example["folder_name"] = "garage"
        example["file_name"] = tgt_frame_id
        example["gt_pose"] = pose
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out
