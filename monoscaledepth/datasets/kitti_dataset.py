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
import torch
import torch.nn.functional as F

from monoscaledepth.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        return os.path.isfile(velo_filename)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits"""
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str
        )
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt,
            self.full_res_shape[::-1],
            order=0,
            preserve_range=True,
            mode="constant",
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIRawPoseDataset(KITTIDataset):
    """KITTI dataset with gt_pose for training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIRawPoseDataset, self).__init__(*args, **kwargs)

        self.load_pose = True

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str
        )
        return image_path

    def get_pose(self, folder, frame_index):
        f_str = "{:010d}.txt".format(frame_index)
        pose_file = os.path.join(self.data_path, folder, "poses", f_str)
        with open(pose_file, "r") as f:
            line = f.readline()
            pose = np.array(line.split(), dtype="float32").reshape((4, 4))
            pose = torch.from_numpy(pose)

        return pose


class KITTIRawPoseSemanticDataset(KITTIRawPoseDataset):
    """KITTI dataset with gt_pose and semantic masks for training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIRawPoseSemanticDataset, self).__init__(*args, **kwargs)

        self.load_semantic = True

    def get_sementic(self, folder, frame_index):
        h, w, max_instances = 192, 640, 5
        masks_file = "{:010d}_masks.npy".format(frame_index)
        masks_path = os.path.join(self.data_path, folder, "semantics", masks_file)
        masks = np.load(masks_path)
        masks = torch.from_numpy(masks)

        # No instance case.
        if masks.shape == (0,):
            masks = torch.zeros((1, h, w))

        # Resize masks.
        masks = F.interpolate(
            masks.unsqueeze(0).float(),
            [h, w],
        )
        masks = masks.squeeze(0)

        # method 1: or all the masks.
        # masks = masks.sum(dim=0, keepdim=True) > 0

        # method 2: fill with zeros up to `max_instances`
        zeros = torch.zeros((max_instances, h, w))
        masks = torch.cat([masks, zeros], dim=0)
        masks = masks[:max_instances]

        return masks


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str,
        )
        return image_path


class KITTIOdomPoseDataset(KITTIDataset):
    """KITTI dataset for odometry with gt_pose for training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIOdomPoseDataset, self).__init__(*args, **kwargs)

        self.load_pose = True

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str,
        )
        return image_path

    def get_pose(self, folder, frame_index):
        pose_file = os.path.join(
            self.data_path,
            "poses/{:02d}.txt".format(int(folder)),
        )
        with open(pose_file, "r") as f:
            lines = f.readlines()
            line = lines[frame_index][:-1] + " 0 0 0 1"
            pose = np.array(line.split(), dtype="float32").reshape((4, 4))
            pose = torch.from_numpy(pose)

        return pose


# class KITTIRawPoseDataset(MonoDataset):
#     """KITTI dataset with gt_pose for training and testing"""

#     RAW_WIDTH = 640
#     RAW_HEIGHT = 192

#     def __init__(self, *args, **kwargs):
#         super(KITTIRawPoseDataset, self).__init__(*args, **kwargs)

#     def index_to_folder_and_frame_idx(self, index):
#         """Convert index in the dataset to a folder name, frame_idx and any other bits

#         txt file is of format:
#             2011_09_26_drive_0001_sync_02 0000000001
#             2011_09_26_drive_0001_sync_02 0000000002
#         """

#         folder, frame_index = self.filenames[index].split()
#         side = None
#         return folder, frame_index, side

#     def get_colors(self, folder, frame_index, side, do_flip):
#         if side is not None:
#             raise ValueError(
#                 "KITTI with gt_pose dataset doesn't know how to deal with sides"
#             )

#         color = self.loader(self.get_image_path(folder, frame_index))
#         color = np.array(color)

#         pose_seq = self.load_pose(folder, frame_index)

#         w = color.shape[1] // 3
#         inputs = {}

#         inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
#         inputs[("color", 0, -1)] = pil.fromarray(color[:, w : 2 * w])
#         inputs[("color", 1, -1)] = pil.fromarray(color[:, 2 * w :])

#         if do_flip:
#             for key in inputs:
#                 inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

#         inputs[("gt_pose", -1)] = pose_seq[0]
#         inputs[("gt_pose", 0)] = pose_seq[1]
#         inputs[("gt_pose", 1)] = pose_seq[2]

#         return inputs

#     def load_intrinsics(self, folder, frame_index):
#         camera_file = os.path.join(
#             self.data_path, folder, "{}_cam.txt".format(frame_index)
#         )
#         camera = np.loadtxt(camera_file, delimiter=",")
#         fx = camera[0]
#         fy = camera[4]
#         u0 = camera[2]
#         v0 = camera[5]
#         intrinsics = np.array(
#             [[fx, 0, u0, 0], [0, fy, v0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#         ).astype(np.float32)

#         intrinsics[0, :] /= self.RAW_WIDTH
#         intrinsics[1, :] /= self.RAW_HEIGHT
#         return intrinsics

#     def load_pose(self, folder, frame_index):
#         pose_file = os.path.join(
#             self.data_path, folder, "{}_pose.txt".format(frame_index)
#         )
#         with open(pose_file, "r") as f:
#             lines = f.readlines()

#         pose_seq = []
#         for line in lines:
#             pose = np.array(line.split(), dtype="float32").reshape((4, 4))
#             pose_seq.append(pose)

#         return pose_seq

#     def check_depth(self):
#         return False

#     def get_image_path(self, folder, frame_index):
#         return os.path.join(self.data_path, folder, "{}.jpg".format(frame_index))
