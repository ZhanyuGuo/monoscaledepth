import os
import numpy as np
from collections import namedtuple

OxtsPacket = namedtuple(
    "OxtsPacket",
    "lat, lon, alt, "
    + "roll, pitch, yaw, "
    + "vn, ve, vf, vl, vu, "
    + "ax, ay, az, af, al, au, "
    + "wx, wy, wz, wf, wl, wu, "
    + "pos_accuracy, vel_accuracy, "
    + "navstat, numsats, "
    + "posmode, velmode, orimode",
)


def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_from_oxts_packet(raw_data, scale):
    """
    Helper method to compute a SE(3) pose matrix from an OXTS packet

    Parameters
    ----------
    raw_data : dict
        Oxts data to read from
    scale : float
        Oxts scale

    Returns
    -------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        Translation vector
    """
    packet = OxtsPacket(*raw_data)
    er = 6378137.0  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def transform_from_rot_trans(R, t):
    """
    Transformation matrix from rotation matrix and translation vector.

    Parameters
    ----------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        translation vector

    Returns
    -------
    matrix : np.array [4,4]
        Transformation matrix
    """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary

    Parameters
    ----------
    filepath : str
        File path to read from

    Returns
    -------
    calib : dict
        Dictionary with calibration values
    """
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


if __name__ == "__main__":
    data_path = "~/dataset/KITTI_RAW/"
    data_path = os.path.expanduser(data_path)
    date_list = os.listdir(data_path)
    for date in date_list:
        date_dir = os.path.join(data_path, date)
        tmp = os.listdir(date_dir)
        drive_list = []
        for t in tmp:
            if "s" in t:
                drive_list.append(t)

        cam2cam = read_calib_file(os.path.join(date_dir, "calib_cam_to_cam.txt"))
        imu2velo = read_calib_file(os.path.join(date_dir, "calib_velo_to_cam.txt"))
        velo2cam = read_calib_file(os.path.join(date_dir, "calib_imu_to_velo.txt"))

        velo2cam_mat = transform_from_rot_trans(velo2cam["R"], velo2cam["T"])
        imu2velo_mat = transform_from_rot_trans(imu2velo["R"], imu2velo["T"])
        cam_2rect_mat = transform_from_rot_trans(cam2cam["R_rect_00"], np.zeros(3))
        
        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        drive_list_length = len(drive_list)
        for i, drive in enumerate(drive_list):
            print("Processing {} ... {}/{}".format(drive, i + 1, drive_list_length))
            dump_path = os.path.join(date_dir, drive, "poses")
            if not os.path.exists(dump_path):
                os.makedirs(dump_path)
            oxts_path = os.path.join(date_dir, drive, "oxts")
            oxts_files = os.listdir(os.path.join(oxts_path, "data"))
            origin_oxts_file = os.path.join(oxts_path, "data", "0000000000.txt")
            origin_oxts_data = np.loadtxt(origin_oxts_file, delimiter=" ", skiprows=0)
            lat = origin_oxts_data[0]
            scale = np.cos(lat * np.pi / 180.0)
            origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
            origin_pose = transform_from_rot_trans(origin_R, origin_t)

            for file in oxts_files:
                oxts_file = os.path.join(oxts_path, "data", file)
                oxts_data = np.loadtxt(oxts_file, delimiter=" ", skiprows=0)
                R, t = pose_from_oxts_packet(oxts_data, scale)
                pose = transform_from_rot_trans(R, t)
                odo_pose = (
                    imu2cam @ np.linalg.inv(origin_pose) @ pose @ np.linalg.inv(imu2cam)
                ).astype(np.float32)
                items = []
                for row in odo_pose:
                    for i in row:
                        items.append(str(i))
                line = " ".join(items)
                with open(os.path.join(dump_path, file), "w") as f:
                    f.write(line)
