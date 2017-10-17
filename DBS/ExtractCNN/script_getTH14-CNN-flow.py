import sys
import os
import scipy.io as sio
import multiprocessing

sys.path.append('.')
from extractCNNbyTSN_flow import build_net, extract_cnn

UCF20Path = '/data/datasets/THUMOS14/TRUE'
UCF20_Feature_Path = '/data3_alpha/datasets/THUMOS14/CNNfeature-flow-kin'


num_worker = 4
layerName = 'inception_5b_output'

def file_path(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

fileList = file_path(UCF20Path)

def process_video(file):
    videoName = file.split('/')[-1].split('.')[0]
    className = file.split('/')[-2]
    outputPath = os.path.join(UCF20_Feature_Path,className)
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    if not os.path.isfile(os.path.join(outputPath,videoName+'.mat')):
        cnn4v = extract_cnn(file, batch_size = 4, layer_name = layerName)
        if cnn4v.shape[0] > 6000:
            sio.savemat(os.path.join(outputPath, videoName + '.mat'), {'cnn4v': cnn4v}, do_compression=False, format='5')
        else:
            sio.savemat(os.path.join(outputPath,videoName+'.mat'), {'cnn4v': cnn4v}, do_compression=True, format='5')
    # else:
    #     cnn4vMat = sio.loadmat(os.path.join(outputPath, videoName + '.mat'))
    #     cnn4v = cnn4vMat['cnn4v']
    #     if cnn4v.shape[0] > 6000:
    #         sio.savemat(os.path.join(outputPath, videoName + '.mat'), {'cnn4v': cnn4v}, do_compression=False, format='5')

    print 'Video %s in class %s is done'%(videoName, className)


if num_worker > 1:
    pool = multiprocessing.Pool(num_worker, initializer=build_net)
    video_scores = pool.map(process_video, fileList)
else:
    build_net()
    video_scores = map(process_video, fileList)