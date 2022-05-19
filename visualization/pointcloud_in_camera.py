import os
import json
import argparse
import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple testing funtion for ManyDepth models."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="path to an image to construct for",
        required=True,
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        help="path to the .npy depth of the image",
        required=True,
    )
    parser.add_argument(
        "--intrinsics_path",
        type=str,
        help="path to a json file containing a normalised 3x3 intrinsics matrix",
        required=True,
    )
    parser.add_argument(
        "--save_path", type=str, help="path to save the .obj", default=""
    )
    parser.add_argument(
        "--crop_beyond",
        type=int,
        help="objects beyond the height will be cropped",
        default=0,
    )

    return parser.parse_args()


def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    with open(intrinsics_path, "r") as f:
        K = np.array(json.load(f))

    K[0, :] *= resize_width
    K[1, :] *= resize_height

    return K


def main(args):
    img = cv.imread(args.image_path, cv.IMREAD_COLOR)

    depth = np.load(args.depth_path)
    depth = depth.squeeze()

    assert (
        img.shape[0] == depth.shape[0] and img.shape[1] == depth.shape[1]
    ), "Not aligned between image and depth."

    # print the min max depth estimation.
    print("Min = {:.3f}, Max = {:.3f}".format(1 / depth.max(), 1 / depth.min()))

    height, width = depth.shape

    point_cloud = []

    K = load_and_preprocess_intrinsics(args.intrinsics_path, width, height)

    for j in range(args.crop_beyond, height):
        for i in range(width):
            ori = np.array(
                [
                    (float(i) - K[0, 2]) / K[0, 0] / depth[j, i],
                    (float(j) - K[1, 2]) / K[1, 1] / depth[j, i],
                    1.0 / depth[j, i],
                    img[j, i, 2] / 255.0,
                    img[j, i, 1] / 255.0,
                    img[j, i, 0] / 255.0,
                ]
            )
            point_cloud.append(ori)

    # output file
    directory, file_name = os.path.split(args.image_path)
    output_name = os.path.splitext(file_name)[0]

    save_path = os.path.join(args.save_path, output_name + ".obj")
    f = open(save_path, "w")
    for point in point_cloud:
        f.write("v")
        for i in range(6):
            f.write(" ")
            f.write(str(point[i]))
        f.write("\n")

    print("Save pointcloud successfully -> {}".format(save_path))


if __name__ == "__main__":
    args = parse_args()
    main(args)
