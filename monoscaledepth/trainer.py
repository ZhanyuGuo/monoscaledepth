# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
from tabnanny import verbose

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
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
)

from . import datasets, networks


_DEPTH_COLORMAP = plt.get_cmap("plasma", 256)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32, e.g. 640x192
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width'  must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1"

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()

        if not self.opt.no_multi_depth:
            # lookup frames in multi frame path
            self.matching_ids = [0]
            if self.opt.use_future_frame:
                self.matching_ids.append(1)

            for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
                self.matching_ids.append(idx)
                if idx not in frames_to_load:
                    frames_to_load.append(idx)

        print("Loading frames:", frames_to_load)

        self.pose_supervise = False

        # fmt: off
        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose or self.opt.no_multi_depth
        # fmt: on

        # MODEL SETUP
        if not self.opt.no_multi_depth:
            # cost volume depth tracker
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
            if self.train_teacher_and_pose:
                print("using adaptive depth binning!")
            else:
                print("fixing pose network and monocular network!")

            # multi depth encoder and decoder
            self.models["encoder"] = networks.ResnetEncoderMatching(
                num_layers=self.opt.num_layers,
                pretrained=self.opt.weights_init == "pretrained",
                input_height=self.opt.height,
                input_width=self.opt.width,
                adaptive_bins=True,
                min_depth_bin=self.min_depth_tracker,
                max_depth_bin=self.max_depth_tracker,
                depth_binning=self.opt.depth_binning,
                num_depth_bins=self.opt.num_depth_bins,
            )
            self.models["depth"] = networks.DepthDecoder(
                num_ch_enc=self.models["encoder"].num_ch_enc, scales=self.opt.scales
            )
            self.models["encoder"].to(self.device)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())

        # mono depth encoder and decoder
        self.models["mono_encoder"] = networks.ResnetEncoder(
            num_layers=18, pretrained=self.opt.weights_init == "pretrained"
        )
        self.models["mono_depth"] = networks.DepthDecoder(
            num_ch_enc=self.models["mono_encoder"].num_ch_enc, scales=self.opt.scales
        )
        self.models["mono_encoder"].to(self.device)
        self.models["mono_depth"].to(self.device)
        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["mono_encoder"].parameters())
            self.parameters_to_train += list(self.models["mono_depth"].parameters())

        # pose encoder and decoder
        self.models["pose_encoder"] = networks.ResnetEncoder(
            num_layers=18,
            pretrained=self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames,
        )
        self.models["pose"] = networks.PoseDecoder(
            num_ch_enc=self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
        )
        self.models["pose_encoder"].to(self.device)
        self.models["pose"].to(self.device)
        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1, verbose=True
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        print("Training model named:", self.opt.model_name)
        print("Models and tensorboard events files are saved to:", self.opt.log_dir)
        print("Training is using:", self.device)

        # DATA
        datasets_dict = {
            "kitti_raw": datasets.KITTIRAWDataset,
            "kitti_raw_pose": datasets.KITTIRawPoseDataset,
            "kitti_raw_pose_semantic": datasets.KITTIRawPoseSemanticDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "kitti_odom_pose": datasets.KITTIOdomPoseDataset,
            # "dominant_pose": datasets.DominantDataset,
        }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)

        # fmt: off
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        # fmt: on

        train_dataset = self.dataset(
            data_path=self.opt.data_path,
            filenames=train_filenames,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=frames_to_load,
            num_scales=self.num_scales,
            is_train=True,
            img_ext=img_ext,
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
        )
        val_dataset = self.dataset(
            data_path=self.opt.data_path,
            filenames=val_filenames,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=frames_to_load,
            num_scales=self.num_scales,
            is_train=False,
            img_ext=img_ext,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]

        print("Using split:", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)
            )
        )

        self.save_opts()

    def train(self):
        """Run the entire training pipeline"""

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if (
                self.opt.add_pose_supervise
                and self.epoch == self.opt.begin_supervise_epoch
            ):
                self.pose_supervise = True

            if (
                self.epoch == self.opt.freeze_teacher_epoch
                and not self.opt.no_multi_depth
            ):
                self.freeze_teacher()

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            print("freezing teacher and pose networks!")

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(
                self.parameters_to_train, self.opt.learning_rate
            )
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1, verbose=True
            )
            # ensure the scheduler having the right step
            for _ in range(self.epoch):
                self.model_lr_scheduler.step()

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation"""

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(
                    batch_idx,
                    duration,
                    losses["loss"].cpu().data,
                    losses["pose"].cpu().data,
                )

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses"""

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs)

        mono_outputs.update(pose_pred)
        outputs.update(pose_pred)

        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models["mono_depth"](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models["mono_depth"](feats))

        self.generate_images_pred(inputs, mono_outputs, is_multi=False)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        if not self.opt.no_multi_depth:
            # fmt: off
            # grab poses + frames and stack for input to the multi frame network
            relative_poses = [inputs[("relative_pose", idx)] for idx in self.matching_ids[1:]]
            relative_poses = torch.stack(relative_poses, 1)

            # batch x frames x 3 x h x w
            lookup_frames = [inputs[("color_aug", idx, 0)] for idx in self.matching_ids[1:]]
            lookup_frames = torch.stack(lookup_frames, 1)

            # apply static frame and zero cost volume augmentation
            batch_size = len(lookup_frames)
            augmentation_mask = (torch.zeros([batch_size, 1, 1, 1]).to(self.device).float())
            # fmt: on
            if is_train and not self.opt.no_matching_augmentation:
                for batch_idx in range(batch_size):
                    rand_num = random.random()
                    # static camera augmentation -> overwrite lookup frames with current frame
                    if rand_num < 0.25:
                        replace_frames = [
                            inputs[("color", 0, 0)][batch_idx]
                            for _ in self.matching_ids[1:]
                        ]
                        replace_frames = torch.stack(replace_frames, 0)
                        lookup_frames[batch_idx] = replace_frames
                        augmentation_mask[batch_idx] += 1
                    elif rand_num < 0.5:
                        relative_poses[batch_idx] *= 0
                        augmentation_mask[batch_idx] += 1
            outputs["augmentation_mask"] = augmentation_mask

            min_depth_bin = self.min_depth_tracker
            max_depth_bin = self.max_depth_tracker

            # update multi frame outputs dictionary with single frame outputs
            for key in list(mono_outputs.keys()):
                _key = list(key)
                if _key[0] in ["depth", "disp"]:
                    _key[0] = "mono_" + key[0]
                    _key = tuple(_key)
                    outputs[_key] = mono_outputs[key]

            # multi frame path
            features, lowest_cost, confidence_mask = self.models["encoder"](
                inputs["color_aug", 0, 0],
                lookup_frames,
                relative_poses,
                inputs[("K", 2)],
                inputs[("inv_K", 2)],
                min_depth_bin=min_depth_bin,
                max_depth_bin=max_depth_bin,
            )
            outputs.update(self.models["depth"](features))

            outputs["lowest_cost"] = F.interpolate(
                lowest_cost.unsqueeze(1),
                [self.opt.height, self.opt.width],
                mode="nearest",
            )[:, 0]
            outputs["consistency_mask"] = F.interpolate(
                confidence_mask.unsqueeze(1),
                [self.opt.height, self.opt.width],
                mode="nearest",
            )[:, 0]

            if not self.opt.disable_motion_masking:
                outputs["consistency_mask"] = outputs[
                    "consistency_mask"
                ] * self.compute_matching_mask(outputs)

            self.generate_images_pred(inputs, outputs, is_multi=True)
            losses = self.compute_losses(inputs, outputs, is_multi=True)

            # update adaptive depth bins
            if self.train_teacher_and_pose:
                self.update_adaptive_depth_bins(outputs)

            # update losses with single frame losses
            if self.train_teacher_and_pose:
                for key, val in mono_losses.items():
                    losses[key] += val
        else:
            losses = mono_losses
            outputs = mono_outputs

        if self.pose_supervise:
            pose_losses = self.compute_pose_losses(inputs, outputs)
        else:
            pose_losses = torch.tensor(0, device=self.device)

        losses["pose"] = pose_losses
        losses["loss"] += self.opt.pose_weight * losses["pose"]

        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[("mono_depth", 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[("mono_depth", 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""

        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {
                f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids
            }
            for f_i in self.opt.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                )

            if not self.opt.no_multi_depth:
                # now we need poses for matching - compute without gradients
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids
                }
                with torch.no_grad():
                    # fmt: off
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in self.matching_ids[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True
                            )

                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, inputs[("relative_pose", fi + 1)])
                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False
                            )

                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, inputs[("relative_pose", fi - 1)])

                        # set missing images to 0 pose
                        for batch_idx, feat in enumerate(pose_feats[fi]):
                            if feat.sum() == 0:
                                pose[batch_idx] *= 0

                        inputs[("relative_pose", fi)] = pose
                    # fmt: on
        else:
            raise NotImplementedError

        return outputs

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp,
                [self.opt.height, self.opt.width],
                mode="bilinear",
                align_corners=False,
            )
            source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True,
                )

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = inputs[
                        ("color", frame_id, source_scale)
                    ]

    def compute_matching_mask(self, outputs):
        """NOTE Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        # ------- Origin -------
        mono_output = outputs[("mono_depth", 0, 0)]
        matching_depth = 1 / outputs["lowest_cost"].unsqueeze(1).to(self.device)

        # ---- Normalization ----
        # mono_output = outputs[("mono_depth", 0, 0)]
        # mono_output_mean = mono_output.mean(2, True).mean(3, True)
        # mono_output /= mono_output_mean + 1e-7

        # matching_depth = 1 / outputs["lowest_cost"].unsqueeze(1).to(self.device)
        # matching_depth_mean = matching_depth.mean(2, True).mean(3, True)
        # matching_depth /= matching_depth_mean + 1e-7

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch"""

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target)
                    )

                identity_reprojection_losses = torch.cat(
                    identity_reprojection_losses, 1
                )

                # differently to Monodepth2, compute mins as we go
                identity_reprojection_loss, _ = torch.min(
                    identity_reprojection_losses, dim=1, keepdim=True
                )
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(
                    reprojection_losses, dim=1, keepdim=True
                )

                # add random numbers to break ties
                identity_reprojection_loss += (
                    torch.randn(identity_reprojection_loss.shape).to(self.device)
                    * 0.00001
                )
            else:
                identity_reprojection_loss = None

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(
                reprojection_loss, identity_reprojection_loss
            )

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = reprojection_loss_mask * outputs[
                        "consistency_mask"
                    ].unsqueeze(1)
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = reprojection_loss_mask * (
                        1 - outputs["augmentation_mask"]
                    )
                consistency_mask = (1 - reprojection_loss_mask).float()

                if self.opt.use_semantic:
                    # # ---- 1 ----
                    # consistency_mask = inputs[("semantic_mask", 0)] # b x 1 x h x w
                    # reprojection_loss_mask = 1 - consistency_mask

                    # ---- 2 ----
                    threshold = 0.5
                    max_instances = 5

                    semantic_masks = inputs[("semantic_mask", 0)]  # b x c x h x w
                    con_and_sem = consistency_mask * semantic_masks
                    and_sum = con_and_sem.sum(dim=[2, 3])
                    sem_sum = semantic_masks.sum(dim=[2, 3]) + 1e-7
                    indices = (and_sum / sem_sum) > threshold
                    # fmt: off
                    for batch_idx in range(self.opt.batch_size):
                        for mask_idx in range(max_instances):
                            if indices[batch_idx, mask_idx]:
                                consistency_mask[batch_idx] = ((consistency_mask[batch_idx] + semantic_masks[batch_idx, mask_idx : mask_idx + 1]) > 0).float()
                            else:
                                consistency_mask[batch_idx] = consistency_mask[batch_idx] * (1 - semantic_masks[batch_idx, mask_idx : mask_idx + 1])
                    # fmt: on
                    reprojection_loss_mask = 1 - consistency_mask

            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (
                reprojection_loss_mask.sum() + 1e-7
            )

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = (
                    torch.abs(multi_depth - mono_depth) * consistency_mask
                )
                consistency_loss = consistency_loss.mean()

                # NOTE: To keep scale aligned with the one before pose supervised.
                if (
                    self.opt.add_pose_supervise
                    and self.epoch >= self.opt.begin_supervise_epoch
                ):
                    consistency_loss /= 30

                # save for logging to tensorboard
                consistency_target = (
                    mono_depth.detach() * consistency_mask
                    + multi_depth.detach() * (1 - consistency_mask)
                )
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses["consistency_loss/{}".format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses["reproj_loss/{}".format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat(
                [reprojection_loss, identity_reprojection_loss], dim=1
            )
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_pose_losses(self, inputs, outputs):
        """Compute pose losses to solve scale ambiguity."""

        pose_losses = 0

        T_W_GT = [inputs[("gt_pose", i)] for i in range(-1, 2)]

        T_n1_W_GT = torch.inverse(T_W_GT[0])
        T_p1_W_GT = torch.inverse(T_W_GT[2])

        T_n1_0_GT = torch.matmul(T_n1_W_GT, T_W_GT[1])
        T_p1_0_GT = torch.matmul(T_p1_W_GT, T_W_GT[1])

        t_n1_0_GT = T_n1_0_GT[:, :3, -1]
        t_p1_0_GT = T_p1_0_GT[:, :3, -1]

        t_n1_0 = outputs[("cam_T_cam", 0, -1)][:, :3, -1]
        t_p1_0 = outputs[("cam_T_cam", 0, 1)][:, :3, -1]

        pose_losses += (t_n1_0_GT.norm(dim=1) - t_n1_0.norm(dim=1)).abs().mean()
        pose_losses += (t_p1_0_GT.norm(dim=1) - t_p1_0.norm(dim=1)).abs().mean()

        return pose_losses

    def val(self):
        """Validate the model on a single minibatch"""

        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""

        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data,
                    self.step,
                )
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data,
                        self.step,
                    )

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image("disp_mono_{}/{}".format(s, j), disp, self.step)

            if not self.opt.no_multi_depth:
                disp = colormap(outputs[("disp", s)][j, 0])
                writer.add_image("disp_multi_{}/{}".format(s, j), disp, self.step)

                if outputs.get("lowest_cost") is not None:
                    lowest_cost = outputs["lowest_cost"][j]

                    consistency_mask = (
                        outputs["consistency_mask"][j]
                        .cpu()
                        .detach()
                        .unsqueeze(0)
                        .numpy()
                    )

                    min_val = np.percentile(lowest_cost.numpy(), 10)
                    max_val = np.percentile(lowest_cost.numpy(), 90)
                    lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                    lowest_cost = colormap(lowest_cost)

                    writer.add_image("lowest_cost/{}".format(j), lowest_cost, self.step)
                    writer.add_image(
                        "lowest_cost_masked/{}".format(j),
                        lowest_cost * consistency_mask,
                        self.step,
                    )
                    writer.add_image(
                        "consistency_mask/{}".format(j), consistency_mask, self.step
                    )

                    consistency_target = colormap(outputs["consistency_target/0"][j])
                    writer.add_image(
                        "consistency_target/{}".format(j), consistency_target, self.step
                    )

    def log_time(self, batch_idx, duration, loss, pose):
        """Print a logging statement to the terminal"""

        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / self.step - 1.0) * time_sofar
            if self.step > 0
            else 0
        )
        print_string = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
            + " | loss: {:.5f} | pose: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                pose,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch)
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == "encoder":
                # save the sizes - these are needed at prediction time
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
                # save estimates of depth bins
                to_save["min_depth_bin"] = self.min_depth_tracker
                to_save["max_depth_bin"] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):
        model_list = ["pose_encoder", "pose", "mono_encoder", "mono_depth"]
        for n in model_list:
            print("loading {}".format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk"""

        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(
            self.opt.load_weights_folder
        ), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == "encoder":
                min_depth_bin = pretrained_dict.get("min_depth_bin")
                max_depth_bin = pretrained_dict.get("max_depth_bin")
                print("min depth", min_depth_bin, "max_depth", max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print("setting depth bins!")
                    self.models["encoder"].compute_depth_bins(
                        min_depth_bin, max_depth_bin
                    )

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def set_train(self):
        """Convert all models to training mode"""

        # for m in self.models.values():
        #     m.train()
        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ["depth", "encoder"]:
                    m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""

        for m in self.models.values():
            m.eval()

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""

        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(to_save, f, indent=2)


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
