import os

img_path = "garage/"
files = os.listdir(img_path)[:]
for file in files:
    file_split = file.split(".")
    frame_id = int(file_split[0])
    ext = file_split[1]
    frame_new_id = frame_id + 10000
    file_new = str(frame_new_id) + "." + ext
    os.rename(img_path + file, img_path + file_new)
    print("{}->{} succeed!".format(file, file_new))
