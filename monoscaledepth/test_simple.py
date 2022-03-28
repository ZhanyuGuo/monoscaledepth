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
        "--image_path",
        type=str,
        help="path to a test image to predict for",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to a folder of weights to load",
        required=True,
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


def test_simple(args):
    """Function to predict for a single image or folder of images"""

    height, width = 192, 640

    assert args.model_path is not None, "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)
    
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
    mono_encoder.eval()
    mono_depth_decoder.eval()
    if torch.cuda.is_available():
        mono_encoder.cuda()
        mono_depth_decoder.cuda()

    # Load input data
    input_image, original_size = load_and_preprocess_image(
        args.image_path,
        resize_width=width,
        resize_height=height,
    )

    with torch.no_grad():
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
        directory, filename = os.path.split(args.image_path)
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

            name_dest_im = os.path.join(
                directory, "{}_{}.jpeg".format(output_name, plot_name)
            )
            im.save(name_dest_im)

            print("-> Saved output image to {}".format(name_dest_im))

    print("-> Done!")


if __name__ == "__main__":
    args = parse_args()
    test_simple(args)
