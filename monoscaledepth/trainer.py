# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from .utils import readlines, sec_to_hm_str
from .layers import (
    SSIM,
    BackprojectDepth,
    Project3D,
    transformation_from_parameters,
    disp_to_depth,
    get_smooth_loss,
    compute_depth_errors,
)

from monoscaledepth import datasets, networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap("plasma", 256)


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join()
        pass
    pass
