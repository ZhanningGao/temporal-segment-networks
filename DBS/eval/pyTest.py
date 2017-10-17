import cv2
import numpy as np

img = cv2.imread('/home/gzn/Desktop/lena.jpeg')

video = cv2.VideoCapture('/data/datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')

try:
    fps = video.get(cv2.cv.CV_CAP_PROP_FRAME_FPS) * 1.0
except:
    fps = 30.0
vData = []

fps_sample = 5.0
step = int(round(fps / fps_sample))

ret, frame = video.read()
vData.append(frame)

p_frame = 0
while ret:
    ret, frame = video.read()
    if p_frame % step == 0:
        vData.append(frame)
        cv2.imshow('test',frame)
        cv2.waitKey(int(1000/fps_sample))
    p_frame += 1

print p_frame, len(vData)
print vData[0].shape[1]
