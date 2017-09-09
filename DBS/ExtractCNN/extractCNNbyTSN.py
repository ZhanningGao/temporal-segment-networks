import argparse
import os
import sys
import math
import cv2

import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

sys.path.append('../..')

caffe_path = '../../lib/caffe-action'

sys.path.append(os.path.join(caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

def build_net():
    net_proto = '../../models/ucf101/tsn_bn_inception_rgb_deploy.prototxt'
    net_weights = '../../models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel'
    num_worker = 2
    gpu_list = [0, 1]
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(net_proto, net_weights, my_id-1)
    else:
        net = CaffeNet(net_proto, net_weights, gpu_list[my_id - 1])

def extract_cnn(videoName, fps_sample = 5.0, batch_size = 1, layer_name = 'inception_5b/output'):
    global net

    cnn4v = []
    # get frame
    video = cv2.VideoCapture(videoName)
    try:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)*1.0
    except:
        fps = 30.0
    vData = []
    step = int(round(fps/fps_sample))

    ret, frame = video.read()
    if frame is not None:
        vData.append(frame)

    p_frame = 1
    while ret:
        ret, frame = video.read()
        if p_frame%step == 0:
            if frame is not None:
                vData.append(frame)
        p_frame+=1

    num_frame = len(vData)

    first = True
    for i in range(0,num_frame,batch_size):
        if i+batch_size <= num_frame:
            inputData = vData[i:i+batch_size]
            featMaps = net.predict_single_frame(inputData, layer_name, frame_size=(298, 224),
                                              over_sample=True, multicrop=False)
            if first:
                cnn4v = featMaps
                first = False
            else:
                cnn4v = np.concatenate((cnn4v,featMaps))

    # last batch
    if num_frame%batch_size:
        inputData = vData[(num_frame//batch_size)*batch_size:num_frame]
        featMaps = net.predict_single_frame(inputData, layer_name, frame_size=(340, 256),
                                        over_sample=True, multicrop=False)
        if first:
            cnn4v = featMaps
        else:
            cnn4v = np.concatenate((cnn4v,featMaps))


    return cnn4v

# build_net()
# import time
# for batch in [1,8,16,32,64]:
#     start = time.time()
#     cnn4v = extract_cnn('/data/datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',batch_size=int(batch))
#     end = time.time()
#     print 'batch = %d, Running times = %.3f'%(batch, end-start)