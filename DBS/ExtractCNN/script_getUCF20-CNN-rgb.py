import sys
import os
import scipy.io as sio
import multiprocessing

sys.path.append('.')
from extractCNNbyTSN import build_net, extract_cnn

UCF20Path = '/data3_alpha/datasets/UCF-20-THUMOS14'
UCF20_Feature_Path = '/data3_alpha/datasets/UCF20/CNNfeature-rgb'


num_worker = 2


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
        cnn4v = extract_cnn(file)
        sio.savemat(os.path.join(outputPath,videoName+'.mat'), {'cnn4v': cnn4v})

    print 'Video %s in class %s is done'%(videoName, className)


if num_worker > 1:
    pool = multiprocessing.Pool(num_worker, initializer=build_net)
    video_scores = pool.map(process_video, fileList)
else:
    build_net()
    video_scores = map(process_video, fileList)