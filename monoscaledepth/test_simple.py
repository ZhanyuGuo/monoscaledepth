# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from monoscaledepth import networks
from .layers import transformation_from_parameters, disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple testing funtion for MonoScaleDepth models."
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
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to a folder of weights to load",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=("multi", "mono"),
        help='"multi" or "mono". If set to "mono" then the network is run without '
        "the source image, e.g. as described in Table 5 of the paper.",
        required=False,
    )
    parser.add_argument(
        "--no_multi_depth",
        action="store_true",
        help="If set, only use mono depth",
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
    K = np.eye(4)
    with open(intrinsics_path, "r") as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK


def test_simple(args):
    """Function to predict for a single image or folder of images"""

    height, width = 192, 640

    assert args.model_path is not None, "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    if not args.no_multi_depth:
        print("   Loading pretrained encoder")
        encoder_dict = torch.load(
            os.path.join(args.model_path, "encoder.pth"), map_location=device
        )
        encoder = networks.ResnetEncoderMatching(
            18,
            False,
            input_width=encoder_dict["width"],
            input_height=encoder_dict["height"],
            adaptive_bins=True,
            min_depth_bin=encoder_dict["min_depth_bin"],
            max_depth_bin=encoder_dict["max_depth_bin"],
            depth_binning="linear",
            num_depth_bins=96,
        )
        filtered_dict_enc = {
            k: v for k, v in encoder_dict.items() if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_dict_enc)

        print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4)
        )
        loaded_dict = torch.load(
            os.path.join(args.model_path, "depth.pth"), map_location=device
        )
        depth_decoder.load_state_dict(loaded_dict)

        print("   Loading pose network")
        pose_enc_dict = torch.load(
            os.path.join(args.model_path, "pose_encoder.pth"), map_location=device
        )
        pose_dec_dict = torch.load(
            os.path.join(args.model_path, "pose.pth"), map_location=device
        )

        pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        pose_dec = networks.PoseDecoder(
            pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
        )

        pose_enc.load_state_dict(pose_enc_dict, strict=True)
        pose_dec.load_state_dict(pose_dec_dict, strict=True)

    print("   Loading pretrained mono encoder")
    mono_encoder_dict = torch.load(
        os.path.join(args.model_path, "mono_encoder.pth"), map_location=device
    )
    mono_encoder = networks.ResnetEncoder(
        num_layers=18,
        pretrained=False,
    )
    mono_encoder.load_state_dict(mono_encoder_dict)

    print("   Loading pretrained mono_decoder")
    mono_decoder_dict = torch.load(
        os.path.join(args.model_path, "mono_depth.pth"), map_location=device
    )
    mono_depth_decoder = networks.DepthDecoder(
        num_ch_enc=mono_encoder.num_ch_enc, scales=range(4)
    )
    mono_depth_decoder.load_state_dict(mono_decoder_dict)

    # Setting states of networks
    if not args.no_multi_depth:
        encoder.eval()
        depth_decoder.eval()
        pose_enc.eval()
        pose_dec.eval()
    mono_encoder.eval()
    mono_depth_decoder.eval()

    if torch.cuda.is_available():
        if not args.no_multi_depth:
            encoder.cuda()
            depth_decoder.cuda()
            pose_enc.cuda()
            pose_dec.cuda()
        mono_encoder.cuda()
        mono_depth_decoder.cuda()

    # Load input data
    input_image, original_size = load_and_preprocess_image(
        args.target_image_path,
        resize_width=width,
        resize_height=height,
    )

    if not args.no_multi_depth:
        source_image, _ = load_and_preprocess_image(
            args.source_image_path,
            resize_width=encoder_dict["width"],
            resize_height=encoder_dict["height"],
        )

        K, invK = load_and_preprocess_intrinsics(
            args.intrinsics_json_path,
            resize_width=encoder_dict["width"],
            resize_height=encoder_dict["height"],
        )

    with torch.no_grad():
        if not args.no_multi_depth:
            # Estimate poses
            pose_inputs = [source_image, input_image]
            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = pose_dec(pose_inputs)
            pose = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=True
            )
            print(pose)

            if args.mode == "mono":
                pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
                source_image *= 0

            # Estimate depth
            output, lowest_cost, _ = encoder(
                current_image=input_image,
                lookup_images=source_image.unsqueeze(1),
                poses=pose.unsqueeze(1),
                K=K,
                invK=invK,
                min_depth_bin=encoder_dict["min_depth_bin"],
                max_depth_bin=encoder_dict["max_depth_bin"],
            )
            output = depth_decoder(output)
        else:
            feats = mono_encoder(input_image)
            output = mono_depth_decoder(feats)

        sigmoid_output = output[("disp", 0)]
        sigmoid_output, _ = disp_to_depth(sigmoid_output, 0.1, 100)
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False
        )
        sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]
        depth = sigmoid_output_resized.squeeze()
        print("Min = {:.3f}, Max = {:.3f}".format(1 / depth.max(), 1 / depth.min()))

        # Saving numpy file
        directory, filename = os.path.split(args.target_image_path)
        output_name = os.path.splitext(filename)[0]
        name_dest_npy = os.path.join(directory, "{}_disp.npy".format(output_name))
        np.save(name_dest_npy, sigmoid_output.cpu().numpy())
        name_dest_npy = os.path.join(
            directory, "{}_disp_resized.npy".format(output_name)
        )
        np.save(name_dest_npy, sigmoid_output_resized)

        # Saving colormapped depth image and cost volume argmin
        for plot_name, toplot in (("disp", sigmoid_output_resized),):
            toplot = toplot.squeeze()
            normalizer = mpl.colors.Normalize(
                vmin=toplot.min(), vmax=np.percentile(toplot, 95)
            )
            mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
            colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            if args.no_multi_depth:
                output_name += "_mono"
            else:
                output_name += "_multi"
            
            name_dest_im = os.path.join(
                directory, "{}_{}.jpeg".format(output_name, plot_name)
            )
            im.save(name_dest_im)

            print("-> Saved output image to {}".format(name_dest_im))

    print("-> Done!")


if __name__ == "__main__":
    args = parse_args()
    test_simple(args)
