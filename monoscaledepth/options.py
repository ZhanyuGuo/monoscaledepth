# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


class MonoscaledepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MonoScaleDepth options")

        # PATHS
        self.parser.add_argument(
            "--data_path",
            type=str,
            help="path to the training data",
            default=os.path.join(file_dir, "kitti_data"),
        )
        self.parser.add_argument(
            "--log_dir",
            type=str,
            help="log directory",
            default=os.path.join(os.path.expanduser("~"), "tmp"),
        )

        # TRAINING options
        self.parser.add_argument(
            "--model_name",
            type=str,
            help="the name of the folder to save the model in",
            default="mdp",
        )
        self.parser.add_argument(
            "--split",
            type=str,
            help="which training split to use",
            choices=[
                "kitti_raw_pose",
            ],
            default="kitti_raw_pose",
        )
        self.parser.add_argument(
            "--height", type=int, help="input image height", default=192
        )
        self.parser.add_argument(
            "--width", type=int, help="input image width", default=640
        )
        self.parser.add_argument(
            "--disparity_smoothness",
            type=float,
            help="disparity smoothness weight",
            default=1e-3,
        )
        self.parser.add_argument(
            "--scales",
            nargs="+",
            type=int,
            help="scales used in the loss",
            default=[0, 1, 2, 3],
        )
        self.parser.add_argument(
            "--min_depth", type=float, help="minimum depth", default=0.1
        )
        self.parser.add_argument(
            "--max_depth", type=float, help="maximum depth", default=100.0
        )
        self.parser.add_argument(
            "--frame_ids",
            nargs="+",
            type=int,
            help="frames to load",
            default=[0, -1, 1],
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            help="dataset to train on",
            default="kitti_raw_pose",
            choices=[
                "kitti_raw_pose",
            ],
        )
        self.parser.add_argument(
            "--png",
            help="if set, trains from raw KITTI png files (instead of jpgs)",
            action="store_true",
        )

        # OPTIMIZATION options
        self.parser.add_argument(
            "--batch_size", type=int, help="batch size", default=12
        )
        self.parser.add_argument(
            "--learning_rate", type=float, help="learning rate", default=1e-4
        )
        self.parser.add_argument(
            "--num_epochs", type=int, help="number of epochs", default=20
        )
        self.parser.add_argument(
            "--scheduler_step_size",
            type=int,
            help="step size of the scheduler",
            default=15,
        )
        
        # ABLATION options
        self.parser.add_argument(
            "--weights_init",
            type=str,
            help="pretrained or scratch",
            default="pretrained",
            choices=["pretrained", "scratch"],
        )

        # SYSTEM options
        self.parser.add_argument(
            "--no_cuda", help="if set disables CUDA", action="store_true"
        )
        self.parser.add_argument(
            "--num_workers", type=int, help="number of dataloader workers", default=12
        )

        # LOGGING options
        self.parser.add_argument(
            "--log_frequency",
            type=int,
            help="number of batches between each tensorboard log",
            default=250,
        )
        self.parser.add_argument(
            "--save_frequency",
            type=int,
            help="number of epochs between each save",
            default=1,
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options