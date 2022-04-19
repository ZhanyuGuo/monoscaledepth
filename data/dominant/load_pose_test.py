import os
import numpy as np
from pprint import pprint as print

# /home/gzy/dataset/dominant/LeftCamera/train/garage
def load_pose(frame_id):
    dataset_dir = "/home/gzy/dataset/dominant/"
    split = "train"
    city = "garage"
    pose_file = os.path.join(dataset_dir, "LeftCamera", split, city, frame_id + ".txt")
    with open(pose_file, "r") as f:
        # line = f.readline().split()
        pose = f.readline()
        # pose = np.array(line, dtype="float").reshape((4, 4))

    return pose

def load_pose2(folder, frame_name):
    data_path = "/home/gzy/dataset/dominant/split_3"
    pose_file = os.path.join(data_path, folder, "{}_pose.txt".format(frame_name))
    with open(pose_file, "r") as f:
        lines = f.readlines()

    pose_seq = []
    for line in lines:
        pose = np.array(line.split(), dtype="float")
        pose_seq.append(pose)

    return pose_seq

# pose = load_pose("10000")
pose = load_pose2("garage", "10001")
print(pose)
