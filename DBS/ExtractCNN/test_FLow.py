import libpydenseflow as DenFlow
import cv2
import numpy as np

frames = []
img1 = cv2.imread('/home/zpengfei/opencv-2.4.13/samples/gpu/basketball1.png')
img2 = cv2.imread('/home/zpengfei/opencv-2.4.13/samples/gpu/basketball2.png')

frames.append(img1.tostring())
frames.append(img2.tostring())
frames.append(img2.tostring())

print img1.shape

denFlow = DenFlow.TVL1FlowExtractor(20)
denFlow.set_device(0)

flow = denFlow.extract_flow(frames,640,480)

flow_img = np.fromstring(flow[0][0],dtype='uint8')
flow_img1 = np.fromstring(flow[1][1],dtype='uint8')

print len(flow)

cv2.imshow('test',flow_img.reshape([480,640]))
cv2.imshow('test1',flow_img1.reshape([480,640]))
cv2.imshow('test2',img1)
cv2.imshow('test3',img2)
cv2.waitKey()
