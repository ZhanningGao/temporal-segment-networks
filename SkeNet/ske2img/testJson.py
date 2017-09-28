import json
import numpy as np
import cv2

with open('/home/zpengfei/code/dataset/UCF-101/pose/HighJump/v_HighJump_g01_c01/v_HighJump_g01_c01_000000000076_pose_keypoints.json') as f:
    keyPoints = json.load(f)

print np.array(keyPoints['people'][0]['pose_keypoints']).reshape([18,3])

def drawHumanPose(keyPoints):
    # number of human
    human_points = []
    for human in keyPoints['people']:
        human_point = np.array(human['pose_keypoints']).reshape([18,3])
        human_points.append(human_point)

    for human_point in human_points:
        pass

lena = cv2.imread('/home/gzn/Desktop/lena.jpeg')
cv2.imshow('test', lena)

cv2.waitKey()