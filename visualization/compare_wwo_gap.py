import os
import numpy as np


if __name__ == "__main__":
    seq, frame_id = 7, 60
    data_path = "~/dataset/KITTI_ODOM/dataset"
    data_path = os.path.expanduser(data_path)
    file_path = os.path.join(data_path, "poses", "%.2d.txt" % seq)
    with open(file_path, "r") as f:
        lines = f.readlines()
        poses = []
        for i in range(frame_id - 1, frame_id + 2):
            # 59 60 61
            # a  b  c
            line = lines[i] + " 0 0 0 1"
            pose = np.array(line.split(), dtype="float32").reshape((4, 4))
            poses.append(pose)
        T_a_c = np.linalg.inv(poses[0]) @ poses[2]
        T_b_c = np.linalg.inv(poses[1]) @ poses[2]

        print("T_b_c:\n", T_b_c)
        print("T_a_c:\n", T_a_c)

# w/o gap 1 frame
# tensor([[[ 1.0000e+00,  2.2339e-03,  7.7483e-04,  1.4084e-02],
#          [-2.2317e-03,  9.9999e-01, -2.9160e-03, -1.3492e-02],
#          [-7.8132e-04,  2.9143e-03,  1.0000e+00,  7.6136e-01],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],
#        device='cuda:0')

# w/o gap 2 frames
# tensor([[[ 1.0000,  0.0047,  0.0025,  0.0241],
#          [-0.0047,  1.0000, -0.0033, -0.0232],
#          [-0.0025,  0.0032,  1.0000,  1.3006],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')

# w/ gap 1 frame
# tensor([[[ 1.0000e+00,  1.7360e-03,  8.3617e-04,  1.2065e-02],
#          [-1.7337e-03,  9.9999e-01, -2.7020e-03, -1.1768e-02],
#          [-8.4084e-04,  2.7005e-03,  1.0000e+00,  6.7762e-01],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],
#        device='cuda:0')

# w/ gap 2 frames
# tensor([[[ 1.0000,  0.0022,  0.0020,  0.0262],
#          [-0.0022,  1.0000, -0.0033, -0.0245],
#          [-0.0021,  0.0033,  1.0000,  1.3453],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
