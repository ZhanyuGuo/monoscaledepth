import os
import shutil


if __name__ == "__main__":
    # src_dir = os.path.expanduser("~/dataset/DOMI_TJDK/")
    # pose_dir = os.path.join(src_dir, "pose")
    # pose_names = os.listdir(pose_dir)
    # for pose in pose_names:
    #     frame_idx = int(pose.split(".")[0])
    #     os.rename(os.path.join(pose_dir, pose), os.path.join(pose_dir, "{:05d}.txt".format(frame_idx + 20000)))


    src_dir = os.path.expanduser("~/dataset/DOMI_TJDK/")
    dst_dir = os.path.expanduser("~/dataset/dominant/raw_data/train/garage")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img_dir = os.path.join(src_dir, "image")

    image_names = os.listdir(img_dir)
    image_names = sorted(image_names)

    i = 0
    for image in image_names:
        frame_idx = image.split(".")[0]
        shutil.copy(os.path.join(img_dir, image), dst_dir)
        shutil.copy(
            os.path.join(src_dir, "pose", "{:05d}.txt".format(int(frame_idx))),
            dst_dir,
        )
        os.rename(
            os.path.join(dst_dir, image),
            os.path.join(dst_dir, "{:05d}.jpg".format(i + 30000)),
        )
        os.rename(
            os.path.join(dst_dir, "{:05d}.txt".format(int(frame_idx))),
            os.path.join(dst_dir, "{:05d}.txt".format(i + 30000)),
        )
        i += 1
