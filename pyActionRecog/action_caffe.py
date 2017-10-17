import sys


import caffe
from caffe.io import oversample
import numpy as np
from utils.io import flow_stack_oversample, fast_list2arr
import cv2

def mirrorsample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    crops_ix = np.empty((1, 4), dtype=int)

    crops_ix[0] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((2 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-1:ix] = crops[ix-1:ix, :, ::-1, :]  # flip for mirrors
    return crops

def flow_stack_mirrorsample(flow_stack, crop_dims):
    """
    This function performs oversampling on flow stacks.
    Adapted from pyCaffe's oversample function
    :param flow_stack: [flow, flow, ...] it is a list of N flow with KxHxW
    :param crop_dims:
    :return:
    """
    im_shape = np.array(flow_stack[0].shape[1:])
    stack_depth = flow_stack[0].shape[0]
    crop_dims = np.array(crop_dims)

    h_center_offset = (im_shape[0] - crop_dims[0])/2
    w_center_offset = (im_shape[1] - crop_dims[1])/2

    crop_ix = np.empty((1, 4), dtype=int)

    crop_ix[0, :] = [h_center_offset, w_center_offset,
                     h_center_offset+crop_dims[0], w_center_offset+crop_dims[1]]

    crop_ix = np.tile(crop_ix, (2,1))

    crops = np.empty((2*len(flow_stack), stack_depth, crop_dims[0], crop_dims[1]),
                     dtype=flow_stack[0].dtype)

    ix = 0
    for flow in flow_stack:
        for crop in crop_ix:
            crops[ix] = flow[:, crop[0]:crop[2], crop[1]:crop[3]]
            ix += 1
        crops[ix-1:ix] = crops[ix-1:ix, :, :, ::-1]  # flip for mirrors
        crops[ix-1:ix, range(0, stack_depth, 2), ...] = 255 - crops[ix-1:ix, range(0, stack_depth, 2), ...]
    return crops


class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None, batch_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        input_shape = self._net.blobs['data'].data.shape

        if input_size is not None:
            input_shape = input_shape[:2] + input_size
            if batch_size is not None:
                input_shape = batch_size + input_shape[1:]

        transformer = caffe.io.Transformer({'data': input_shape})

        if self._net.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        else:
            pass # non RGB data need not use transformer

        self._transformer = transformer

        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, multicrop=True):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                if multicrop:
                    os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
                else:
                    os_frame = mirrorsample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = [cv2.resize(x.transpose(1,2,0), frame_size) for x in frame]
            frame = [x.transpose(2,0,1) for x in frame]

        if over_sample:
            os_frame = flow_stack_mirrorsample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr(frame)

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()


