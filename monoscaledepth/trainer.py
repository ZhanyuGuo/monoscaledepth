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
)

from monoscaledepth import datasets, networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap("plasma", 256)


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
        frames_to_load = self.opt.frame_ids.copy()  ## [0, -1, 1]

        print("Loading frames: {}".format(frames_to_load))

        self.pose_supervise = False

        # MODEL SETUP
        # depth encoder and decoder
        self.models["mono_encoder"] = networks.ResnetEncoder(
            num_layers=18, pretrained=self.opt.weights_init == "pretrained"
        )
        self.models["mono_depth"] = networks.DepthDecoder(
            num_ch_enc=self.models["mono_encoder"].num_ch_enc, scales=self.opt.scales
        )
        self.models["mono_encoder"].to(self.device)
        self.models["mono_depth"].to(self.device)
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
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )

        print("Training model named:\n", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n", self.opt.log_dir)
        print("Training is using:\n", self.device)

        # DATA
        datasets_dict = {
            "kitti_raw_pose": datasets.KITTIRawPoseDataset,
        }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = (
            num_train_samples // self.opt.batch_size * self.opt.num_epochs
        )

        train_dataset = self.dataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            frames_to_load,
            self.num_scales,
            is_train=True,
            img_ext=img_ext,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            frames_to_load,
            self.num_scales,
            is_train=False,
            img_ext=img_ext,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
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

        print("Using split:\n", self.opt.split)
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

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

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

        pose_pred = self.predict_poses(inputs)
        mono_outputs.update(pose_pred)

        feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
        mono_outputs.update(self.models["mono_depth"](feats))

        self.generate_images_pred(inputs, mono_outputs, is_multi=False)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        if self.pose_supervise:
            pose_losses = self.compute_pose_losses(inputs, mono_outputs)
        else:
            pose_losses = torch.tensor(0)

        mono_losses["pose"] = pose_losses
        mono_losses["loss"] += self.opt.pose_weight * mono_losses["pose"]

        return mono_outputs, mono_losses

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

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                # differently to Monodepth2, compute mins as we go
                identity_reprojection_loss, _ = torch.min(
                    identity_reprojection_losses, dim=1, keepdim=True
                )
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

                # add random numbers to break ties
                identity_reprojection_loss += (
                    torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                )
            else:
                identity_reprojection_loss = None

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(
                reprojection_loss, identity_reprojection_loss
            )

            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (
                reprojection_loss_mask.sum() + 1e-7
            )

            losses["reproj_loss/{}".format(scale)] = reprojection_loss
            loss += reprojection_loss

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

    def log_time(self, batch_idx, duration, loss):
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
            + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
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

    def set_train(self):
        """Convert all models to training mode"""

        for m in self.models.values():
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
