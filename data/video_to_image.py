import os
import cv2


def video2img(video_path, save_dir, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    suc = cap.isOpened()

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = frame_num / fps

    print("Duration of the video: {:.2f} seconds".format(dur))
    print("Number of frames: {}".format(frame_num))
    print("Frames per second (FPS): {}".format(fps))
    print("Total ouput: {}".format(frame_num // frame_interval))

    frame_count = 0
    while suc:
        suc, frame = cap.read()
        if suc and frame_count % frame_interval == 0 and frame_count >= 100 and frame_count <= 11999:
            frame = frame[84 : 360 - 84]
            save_path = os.path.join(save_dir, "{:05d}.jpg".format(frame_count + 30000))
            cv2.imwrite(save_path, frame)
            print(frame_count, suc)
        frame_count += 1

    cap.release()


if __name__ == "__main__":
    video_path = "/media/new_data2/dataset/DOMI_TJDK/LeftCamera01.avi"
    image_dir = "/media/new_data2/dataset/DOMI_TJDK/image"
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    video2img(video_path, image_dir)
