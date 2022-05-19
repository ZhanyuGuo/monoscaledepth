# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from torchvision import transforms

from monoscaledepth import networks
from .layers import (
    BackprojectDepth,
    Project3D,
    SSIM,
    transformation_from_parameters,
    disp_to_depth,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization on something.")

    parser.add_argument(
        "--model_path",
        type=str,
        help="path to a folder of weights to load",
        required=True,
    )
    parser.add_argument(
        "--target_image_path",
        type=str,
        help="path to a test image to predict for",
        required=True,
    )
    parser.add_argument(
        "--source_image_path",
        type=str,
        help="path to a previous image in the video sequence",
        required=False,
    )
    parser.add_argument(
        "--intrinsics_json_path",
        type=str,
        help="path to a json file containing a normalised 3x3 intrinsics matrix",
        required=False,
    )
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K_identity = np.eye(4)
    with open(intrinsics_path, "r") as f:
        K_identity[:3, :3] = np.array(json.load(f))

    K_list = []
    invK_list = []
    for scale in range(4):
        K = K_identity.copy()

        K[0, :] *= resize_width // (2 ** scale)
        K[1, :] *= resize_height // (2 ** scale)

        invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
        K = torch.Tensor(K).unsqueeze(0)
        if torch.cuda.is_available():
            K = K.cuda()
            invK = invK.cuda()

        K_list.append(K)
        invK_list.append(invK)

    return K_list, invK_list


def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images"""

    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim = SSIM()
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
    """Compute loss masks for each of standard reprojection and depth hint reprojection"""

    if identity_reprojection_loss is None:
        # we are not using automasking - standard reprojection loss applied to all pixels
        reprojection_loss_mask = torch.ones_like(reprojection_loss)

    else:
        # we are using automasking
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        reprojection_loss_mask = (idxs == 0).float()

    return reprojection_loss_mask


def compute_matching_mask(outputs, device):
    """NOTE Generate a mask of where we cannot trust the cost volume, based on the difference
    between the cost volume and the teacher, monocular network"""

    # ------- Origin -------
    mono_output = outputs[("mono_depth", 0, 0)]
    matching_depth = 1 / outputs["lowest_cost"].unsqueeze(1).to(device)

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


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    print("-> Loading mono encoder")
    mono_encoder_dict = torch.load(
        os.path.join(args.model_path, "mono_encoder.pth"), map_location=device
    )
    mono_encoder = networks.ResnetEncoder(
        num_layers=18,
        pretrained=False,
    )
    mono_encoder.load_state_dict(mono_encoder_dict)

    print("-> Loading mono decoder")
    mono_decoder_dict = torch.load(
        os.path.join(args.model_path, "mono_depth.pth"), map_location=device
    )
    mono_decoder = networks.DepthDecoder(
        num_ch_enc=mono_encoder.num_ch_enc, scales=range(4)
    )
    mono_decoder.load_state_dict(mono_decoder_dict)

    print("-> Loading multi encoder")
    multi_encoder_dict = torch.load(
        os.path.join(args.model_path, "encoder.pth"), map_location=device
    )
    multi_encoder = networks.ResnetEncoderMatching(
        18,
        False,
        input_width=multi_encoder_dict["width"],
        input_height=multi_encoder_dict["height"],
        adaptive_bins=True,
        min_depth_bin=multi_encoder_dict["min_depth_bin"],
        max_depth_bin=multi_encoder_dict["max_depth_bin"],
        depth_binning="linear",
        num_depth_bins=96,
    )
    filtered_dict = {
        k: v for k, v in multi_encoder_dict.items() if k in multi_encoder.state_dict()
    }
    multi_encoder.load_state_dict(filtered_dict)

    print("-> Loading multi decoder")
    multi_decoder_dict = torch.load(
        os.path.join(args.model_path, "depth.pth"), map_location=device
    )
    multi_decoder = networks.DepthDecoder(
        num_ch_enc=multi_encoder.num_ch_enc, scales=range(4)
    )
    multi_decoder.load_state_dict(multi_decoder_dict)

    print("-> Loading pose encoder")
    pose_encoder_dict = torch.load(
        os.path.join(args.model_path, "pose_encoder.pth"), map_location=device
    )
    pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_encoder.load_state_dict(pose_encoder_dict, strict=True)

    print("-> Loading pose decoder")
    pose_decoder_dict = torch.load(
        os.path.join(args.model_path, "pose.pth"), map_location=device
    )
    pose_decoder = networks.PoseDecoder(
        pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
    )
    pose_decoder.load_state_dict(pose_decoder_dict, strict=True)

    mono_encoder.eval()
    mono_decoder.eval()
    multi_encoder.eval()
    multi_decoder.eval()
    pose_encoder.eval()
    pose_decoder.eval()
    if torch.cuda.is_available():
        mono_encoder.cuda()
        mono_decoder.cuda()
        multi_encoder.cuda()
        multi_decoder.cuda()
        pose_encoder.cuda()
        pose_decoder.cuda()

    target_image, original_size = load_and_preprocess_image(
        args.target_image_path,
        resize_width=multi_encoder_dict["width"],
        resize_height=multi_encoder_dict["height"],
    )
    source_image, _ = load_and_preprocess_image(
        args.source_image_path,
        resize_width=multi_encoder_dict["width"],
        resize_height=multi_encoder_dict["height"],
    )
    K, invK = load_and_preprocess_intrinsics(
        args.intrinsics_json_path,
        resize_width=multi_encoder_dict["width"],
        resize_height=multi_encoder_dict["height"],
    )

    with torch.no_grad():
        # Mono depth
        mono_output = mono_encoder(target_image)
        mono_output = mono_decoder(mono_output)
        mono_sigmoid_output = mono_output[("disp", 0)]
        mono_sigmoid_output, mono_depth = disp_to_depth(mono_sigmoid_output, 0.1, 100)

        # Pose
        pose_inputs = [source_image, target_image]
        pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]
        axisangle, translation = pose_decoder(pose_inputs)
        pose = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True
        )

        # Multi
        multi_output, lowest_cost, confidence_mask = multi_encoder(
            current_image=target_image,
            lookup_images=source_image.unsqueeze(1),
            poses=pose.unsqueeze(1),
            K=K[2],
            invK=invK[2],
            min_depth_bin=multi_encoder_dict["min_depth_bin"],
            max_depth_bin=multi_encoder_dict["max_depth_bin"],
        )
        multi_output = multi_decoder(multi_output)
        multi_output["lowest_cost"] = F.interpolate(
            lowest_cost.unsqueeze(1),
            [multi_encoder_dict["height"], multi_encoder_dict["width"]],
            mode="nearest",
        )[:, 0]
        multi_output["consistency_mask"] = F.interpolate(
            confidence_mask.unsqueeze(1),
            [multi_encoder_dict["height"], multi_encoder_dict["width"]],
            mode="nearest",
        )[:, 0]
        multi_output[("mono_depth", 0, 0)] = mono_depth
        multi_output["consistency_mask"] = multi_output[
            "consistency_mask"
        ] * compute_matching_mask(multi_output, device)

        multi_sigmoid_output = multi_output[("disp", 0)]
        multi_sigmoid_output, multi_depth = disp_to_depth(
            multi_sigmoid_output, 0.1, 100
        )

        # Warp init
        backproject_depth = BackprojectDepth(
            batch_size=1,
            height=multi_encoder_dict["height"],
            width=multi_encoder_dict["width"],
        )
        project_3d = Project3D(
            batch_size=1,
            height=multi_encoder_dict["height"],
            width=multi_encoder_dict["width"],
        )
        backproject_depth.to(device)
        project_3d.to(device)

        # Mono warped
        cam_points = backproject_depth(mono_depth, invK[0])
        pix_coords = project_3d(cam_points, K[0], pose)
        mono_warped = F.grid_sample(
            source_image,
            pix_coords,
            padding_mode="border",
            # padding_mode="zeros",
            align_corners=True,
        )

        # Multi warped
        cam_points = backproject_depth(multi_depth, invK[0])
        pix_coords = project_3d(cam_points, K[0], pose)
        multi_warped = F.grid_sample(
            source_image,
            pix_coords,
            padding_mode="border",
            # padding_mode="zeros",
            align_corners=True,
        )

        mono_reprojection_losses = []
        mono_reprojection_losses.append(
            compute_reprojection_loss(mono_warped, target_image)
        )
        mono_reprojection_losses = torch.cat(mono_reprojection_losses, 1)

        identity_reprojection_losses = []
        identity_reprojection_losses.append(
            compute_reprojection_loss(source_image, target_image)
        )
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        mono_reprojection_loss_mask = compute_loss_masks(
            mono_reprojection_losses, identity_reprojection_losses
        )

        multi_reprojection_loss_mask = torch.ones_like(mono_reprojection_losses)
        multi_reprojection_loss_mask = multi_reprojection_loss_mask * multi_output[
            "consistency_mask"
        ].unsqueeze(1)
        consistency_mask = 1 - multi_reprojection_loss_mask

        mono_sigmoid_output_resized = torch.nn.functional.interpolate(
            mono_sigmoid_output, original_size, mode="bilinear", align_corners=False
        )
        mono_sigmoid_output_resized = mono_sigmoid_output_resized.cpu().numpy()[:, 0]
        mono_disp = mono_sigmoid_output_resized.squeeze()
        print(
            "Min = {:.3f}, Max = {:.3f}".format(
                1 / mono_disp.max(), 1 / mono_disp.min()
            )
        )

        multi_sigmoid_output_resized = torch.nn.functional.interpolate(
            multi_sigmoid_output, original_size, mode="bilinear", align_corners=False
        )
        multi_sigmoid_output_resized = multi_sigmoid_output_resized.cpu().numpy()[:, 0]
        multi_disp = multi_sigmoid_output_resized.squeeze()
        print(
            "Min = {:.3f}, Max = {:.3f}".format(
                1 / multi_disp.max(), 1 / multi_disp.min()
            )
        )

        print(
            "min_depth_bin = {:.3f}, max_depth_bin = {:.3f}".format(
                multi_encoder_dict["min_depth_bin"], multi_encoder_dict["max_depth_bin"]
            )
        )

        directory, filename = os.path.split(args.target_image_path)
        output_name = os.path.splitext(filename)[0]

        mono_reprojection_loss_mask = (
            mono_reprojection_loss_mask.squeeze().cpu().numpy()
        )
        im = pil.fromarray((mono_reprojection_loss_mask * 255).astype(np.uint8))
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "mono_mask")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        consistency_mask = consistency_mask.squeeze().cpu().numpy()
        im = pil.fromarray((consistency_mask * 255).astype(np.uint8))
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "consistency_mask")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        mono_warped_resized = torch.nn.functional.interpolate(
            mono_warped, original_size, mode="bilinear", align_corners=False
        )
        mono_warped_resized = mono_warped_resized.squeeze().cpu().numpy()
        mono_warped_resized = mono_warped_resized.transpose((1, 2, 0))
        im = pil.fromarray((mono_warped_resized * 255).astype(np.uint8))
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "mono_warped")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        multi_warped_resized = torch.nn.functional.interpolate(
            multi_warped, original_size, mode="bilinear", align_corners=False
        )
        multi_warped_resized = multi_warped_resized.squeeze().cpu().numpy()
        multi_warped_resized = multi_warped_resized.transpose((1, 2, 0))
        im = pil.fromarray((multi_warped_resized * 255).astype(np.uint8))
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "multi_warped")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        normalizer = mpl.colors.Normalize(
            vmin=mono_disp.min(), vmax=np.percentile(mono_disp, 95)
        )
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(mono_disp)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "mono_disp")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        normalizer = mpl.colors.Normalize(
            vmin=multi_disp.min(), vmax=np.percentile(multi_disp, 95)
        )
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(multi_disp)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        name_dest_im = os.path.join(
            directory, "{}_{}_full.jpeg".format(output_name, "multi_disp")
        )
        im.save(name_dest_im)
        print("-> Saved output image to {}".format(name_dest_im))

        name_dest_npy = os.path.join(
            directory, "{}_mono_disp_resized.npy".format(output_name)
        )
        np.save(name_dest_npy, mono_sigmoid_output_resized)
        print("-> Saved npy file to {}".format(name_dest_npy))

        name_dest_npy = os.path.join(
            directory, "{}_multi_disp_resized.npy".format(output_name)
        )
        np.save(name_dest_npy, multi_sigmoid_output_resized)
        print("-> Saved npy file to {}".format(name_dest_npy))

        name_dest_npy = os.path.join(
            directory, "{}_consistency.npy".format(output_name)
        )
        np.save(name_dest_npy, consistency_mask)
        print("-> Saved npy file to {}".format(name_dest_npy))


if __name__ == "__main__":
    args = parse_args()
    main(args)
