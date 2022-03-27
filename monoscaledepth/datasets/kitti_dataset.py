# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import skimage.transform
import numpy as np
import PIL.Image as pil

from monoscaledepth.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIRawPoseDataset(MonoDataset):
    """KITTI dataset with gt_pose for training and testing"""
    
    RAW_WIDTH = 640
    RAW_HEIGHT = 192

    def __init__(self, *args, **kwargs):
        super(KITTIRawPoseDataset, self).__init__(*args, **kwargs)
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            2011_09_26_drive_0001_sync_02 0000000001
            2011_09_26_drive_0001_sync_02 0000000002
        """
        folder, frame_index = self.filenames[index].split()
        side = None
        return folder, frame_index, side

    def get_colors(self, folder, frame_index, side, do_flip):
        if side is not None:
            raise ValueError(
                "KITTI with gt_pose dataset doesn't know how to deal with sides"
            )
        
        color = self.loader(self.get_image_path(folder, frame_index))
        color = np.array(color)

        w = color.shape[1] // 3
        inputs = {}

        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w : 2 * w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2 * w :])

        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

        return inputs

    def load_intrinsics(self, folder, frame_index):
        camera_file = os.path.join(
            self.data_path, folder, "{}_cam.txt".format(frame_index)
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

    def check_depth(self):
        return False
    
    def get_image_path(self, folder, frame_index):
        return os.path.join(self.data_path, folder, "{}.jpg".format(frame_index))
