# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from .utils import readlines
from .options import MonoscaledepthOptions
from monoscaledepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = "splits"


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)
    
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(
        opt.load_weights_folder
    )
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Setup dataloaders
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    if opt.no_multi_depth:
        encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
        encoder_class = networks.ResnetEncoder
    else:
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_class = networks.ResnetEncoderMatching

    encoder_dict = torch.load(encoder_path)

    try:
        HEIGHT, WIDTH = encoder_dict["height"], encoder_dict["width"]
    except KeyError:
        print(
            'No "height" or "width" keys found in the encoder state_dict, resorting to '
            "using command line values!"
        )
        HEIGHT, WIDTH = opt.height, opt.width

    dataset = datasets.KITTIRAWDataset(
        opt.data_path, filenames, HEIGHT, WIDTH, frames_to_load, 4, is_train=False
    )
    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if opt.no_multi_depth:
        encoder_opts = dict(num_layers=opt.num_layers, pretrained=False)
    else:
        encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=10.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins)
        pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
        pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

        pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

        pose_enc.load_state_dict(pose_enc_dict, strict=True)
        pose_dec.load_state_dict(pose_dec_dict, strict=True)

        min_depth_bin = encoder_dict.get('min_depth_bin')
        max_depth_bin = encoder_dict.get('max_depth_bin')

        pose_enc.eval()
        pose_dec.eval()

        if torch.cuda.is_available():
            pose_enc.cuda()
            pose_dec.cuda()

    encoder = encoder_class(**encoder_opts)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.eval()
    depth_decoder.eval()

    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            input_color = data[("color", 0, 0)]
            if torch.cuda.is_available():
                input_color = input_color.cuda()
            
            if opt.no_multi_depth:
                output = encoder(input_color)
                output = depth_decoder(output)
            else:
                if opt.static_camera:
                    for f_i in frames_to_load:
                        data["color", f_i, 0] = data[('color', 0, 0)]

                # predict poses
                pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
                if torch.cuda.is_available():
                    pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in frames_to_load[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                        axisangle, translation = pose_dec(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                        axisangle, translation = pose_dec(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                    data[('relative_pose', fi)] = pose

                lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
                lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                relative_poses = torch.stack(relative_poses, 1)

                K = data[('K', 2)]  # quarter resolution for matching
                invK = data[('inv_K', 2)]

                if torch.cuda.is_available():
                    lookup_frames = lookup_frames.cuda()
                    relative_poses = relative_poses.cuda()
                    K = K.cuda()
                    invK = invK.cuda()

                if opt.zero_cost_volume:
                    relative_poses *= 0

                if opt.post_process:
                    raise NotImplementedError

                output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                        relative_poses,
                                                        K,
                                                        invK,
                                                        min_depth_bin, max_depth_bin)
                output = depth_decoder(output)
            pred_disp, _ = disp_to_depth(
                output[("disp", 0)], opt.min_depth, opt.max_depth
            )
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    print("finished predicting!")

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(
        gt_path, fix_imports=True, encoding="latin1", allow_pickle=True
    )["data"]
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")
    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array(
            [
                0.40810811 * gt_height,
                0.99189189 * gt_height,
                0.03594771 * gt_width,
                0.96405229 * gt_width,
            ]
        ).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(
        " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
            med, np.std(ratios / med)
        )
    )
    mean_errors = np.array(errors).mean(0)

    print(
        "\n  "
        + ("{:>8} | " * 7).format(
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        )
    )
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonoscaledepthOptions()
    evaluate(options.parse())
