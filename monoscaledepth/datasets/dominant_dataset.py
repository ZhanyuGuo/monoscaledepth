# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class DominantDataset(MonoDataset):
    """Dominant dataset"""

    RAW_WIDTH = 640
    RAW_HEIGHT = 192

    def __init__(self, *args, **kwargs):
        super(DominantDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            garage 10000
            garage 10001
        """
        folder, frame_id = self.filenames[index].split()
        side = None
        return folder, frame_id, side

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner

        camera_file = os.path.join(
            self.data_path, city, "{}_cam.txt".format(frame_name)
        )
        camera = np.loadtxt(camera_file, delimiter=",")
        fx = camera[0]
        fy = camera[4]
        u0 = camera[2]
        v0 = camera[5]
        intrinsics = np.array(
            [[fx, 0, u0, 0], [0, fy, v0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ).astype(np.float32)

        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def get_pose(self, folder, frame_name):
        pose_file = os.path.join(
            self.data_path, folder, "{}_pose.txt".format(frame_name)
        )
        with open(pose_file, "r") as f:
            lines = f.readlines()

        pose_seq = []
        for line in lines:
            pose = np.array(line.split(), dtype="float32").reshape((4, 4))
            pose_seq.append(pose)

        return pose_seq

    def get_colors(self, folder, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Dominant dataset doesn't know how to deal with sides")

        color = self.loader(self.get_image_path(folder, frame_name))
        color = np.array(color)

        pose_seq = self.get_pose(folder, frame_name)  # length = 3   -1 0 1

        w = color.shape[1] // 3
        inputs = {}

        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w : 2 * w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2 * w :])

        # w = color.shape[1] // 5
        # inputs[("color", -2, -1)] = pil.fromarray(color[:, :w])
        # inputs[("color", -1, -1)] = pil.fromarray(color[:, w : 2 * w])
        # inputs[("color", 0, -1)] = pil.fromarray(color[:, 2 * w : 3 * w])
        # inputs[("color", 1, -1)] = pil.fromarray(color[:, 3 * w : 4 * w])
        # inputs[("color", 2, -1)] = pil.fromarray(color[:, 4 * w :])

        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

        inputs[("gt_pose", -1)] = pose_seq[0]
        inputs[("gt_pose", 0)] = pose_seq[1]
        inputs[("gt_pose", 1)] = pose_seq[2]

        return inputs

    def get_image_path(self, folder, frame_name):
        return os.path.join(self.data_path, folder, "{}.jpg".format(frame_name))
