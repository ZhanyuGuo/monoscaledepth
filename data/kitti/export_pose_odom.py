import os


if __name__ == "__main__":
    data_path = "~/dataset/KITTI_ODOM/dataset"
    data_path = os.path.expanduser(data_path)
    pose_folder = os.path.join(data_path, "poses")
    print("Exporting pose from {}.".format(pose_folder))
    pose_files = os.listdir(pose_folder)
    for pose_file in pose_files:
        with open(os.path.join(pose_folder, pose_file), "r") as f:
            seq = pose_file.split(".")[0]
            print("processing {} ...".format(seq))
            dump_path = os.path.join(data_path, "sequences", seq, "image_2")
            if not os.path.exists(dump_path):
                os.makedirs(dump_path)

            lines = f.readlines()
            for idx, line in enumerate(lines):
                with open(os.path.join(dump_path, "%.6d.txt" % idx), "w") as v:
                    line = line[:-1] + " 0 0 0 1"
                    v.write(line)
