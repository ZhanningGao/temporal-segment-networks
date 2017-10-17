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
import libpydenseflow as DenFlow

def build_net():
    net_proto = '/home/gzn/code/repository/temporal-segment-networks/models/bn_inception_kinetics_flow_pretrained/bn_inception_flow_deploy.prototxt'
    net_weights = '/home/gzn/code/repository/temporal-segment-networks/models/bn_inception_kinetics_flow_pretrained/bn_inception_kinetics_flow_pretrained.caffemodel'
    num_worker = 4
    gpu_list = [0,1,2,3]
    global net
    global denFlow

    my_id = multiprocessing.current_process()._identity[0] \
        if num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(net_proto, net_weights, my_id-1)
        denFlow = DenFlow.TVL1FlowExtractor(20)
        denFlow.set_device(my_id-1)
    else:
        net = CaffeNet(net_proto, net_weights, gpu_list[my_id - 1])
        denFlow = DenFlow.TVL1FlowExtractor(20)
        denFlow.set_device(gpu_list[my_id - 1])

def extract_cnn(videoName, fps_sample = 5.0, batch_size = 1, layer_name = 'inception_5b_output', frame_max = None, flow_len = 5):
    global net
    global denFlow

    cnn4v = []
    # get frame
    video = cv2.VideoCapture(videoName)
    try:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)*1.0
    except:
        fps = 30.0
    vData_all = []
    p_frame = 1

    flow_h = 340
    flow_w = 256

    ret, frame = video.read()
    if frame is not None:
        vData_all.append(cv2.resize(frame, (flow_h,flow_w)).tostring())
    while ret:
        ret, frame = video.read()
        if frame is not None:
            vData_all.append(cv2.resize(frame, (flow_h,flow_w)).tostring())
        if frame_max is not None:
            if p_frame >= frame_max:
                break
        p_frame+=1



    flow_all = denFlow.extract_flow(vData_all, flow_h, flow_w)

    vFlow = []
    for flow in flow_all:
        vFlow.append(np.fromstring(flow[0],dtype='uint8').reshape([flow_w,flow_h]))
        vFlow.append(np.fromstring(flow[1], dtype='uint8').reshape([flow_w,flow_h]))

    num_flow = len(vFlow)

    vFlow = fast_list2arr(vFlow)
    step = int(round(fps/fps_sample))

    flow_stack = 5

    vData = []
    for i_frame in range(0,num_flow, step*2):
        if i_frame+flow_stack*2 <= num_flow:
            vData.append(vFlow[i_frame:i_frame+flow_stack*2])
        else:
            offset = (i_frame+flow_stack*2 - num_flow)/2
            vFlow_extra = np.tile(vFlow[-2:],(offset,1,1))
            vData.append(np.append(vFlow[i_frame:],vFlow_extra,axis=0))

    num_frame = len(vData)
    first = True
    for i in range(0,num_frame,batch_size):
        if i+batch_size <= num_frame:
            inputData = vData[i:i+batch_size]
            featMaps = net.predict_flow_stack(inputData, layer_name, frame_size=(298, 224),
                                              over_sample=True)
            if first:
                cnn4v = featMaps
                first = False
            else:
                cnn4v = np.concatenate((cnn4v,featMaps))

    # last batch
    if num_frame%batch_size:
        inputData = vData[(num_frame//batch_size)*batch_size:num_frame]
        featMaps = net.predict_flow_stack(inputData, layer_name, frame_size=(298, 224),
                                        over_sample=True)
        if first:
            cnn4v = featMaps
        else:
            cnn4v = np.concatenate((cnn4v,featMaps))


    return cnn4v

def fast_list2arr(data, offset=None, dtype=None):
        """
        Convert a list of numpy arrays with the same size to a large numpy array.
        This is way more efficient than directly using numpy.array()
        See
            https://github.com/obspy/obspy/wiki/Known-Python-Issues
        :param data: [numpy.array]
        :param offset: array to be subtracted from the each array.
        :param dtype: data type
        :return: numpy.array
        """
        num = len(data)
        out_data = np.empty((num,) + data[0].shape, dtype=dtype if dtype else data[0].dtype)
        for i in xrange(num):
            out_data[i] = data[i] - offset if offset else data[i]
        return out_data

# build_net()
# import time
# for batch in [8]:
#     start = time.time()
#     cnn4v = extract_cnn('/data/datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',batch_size=int(batch))
#     end = time.time()
#     print 'batch = %d, Running times = %.3f'%(batch, end-start)