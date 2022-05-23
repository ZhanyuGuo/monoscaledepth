import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


def read_obj(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        n = len(lines)
        points = np.zeros((n, 3))
        for i, line in enumerate(lines):
            line = line.split()
            for j in range(3):
                points[i, j] = line[j + 1]

    return points


def main():
    grid_h = 640
    y_min, y_max = -0.5, 1.5
    z_min, z_max = 2, 30

    points = read_obj("10563.obj")

    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    print(point_min, point_max)

    point_w = point_max[0] - point_min[0]
    point_h = z_max - z_min
    print(point_w, point_h)

    grid_w = math.ceil(grid_h * point_w / point_h)
    print(grid_h, grid_w)

    rel_w = grid_w / (point_w * 1.1)
    rel_h = grid_h / (point_h * 1.1)
    print(rel_w, rel_h)

    img = np.zeros((grid_h, grid_w, 3), np.uint8)
    img.fill(200)

    for point in points:
        x, y, z = point[0], point[1], point[2]
        if y > y_min and y < y_max and z > z_min and z < z_max:
            x = (x - point_min[0]) * rel_w
            z = grid_h - (z - z_min) * rel_h
            img[int(z), int(x), :] = 0

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    cv2.imwrite("grid.jpg", img)


if __name__ == "__main__":
    main()
