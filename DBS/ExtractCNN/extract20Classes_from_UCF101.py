# extract 20 classes from UCF-101 dataset to a new folder
import os
import shutil

gtpath = '../data/TH14evalkit/groundtruth'

UCF101Path = '/data3_alpha/datasets/UCF-101'
UCF20Path  = '/data3_alpha/datasets/UCF-20-THUMOS14'

detClassFile = open(os.path.join(gtpath, 'detclasslist.txt'))

classNames = dict()

for line in detClassFile.readlines():
    classNames[(line.split()[-1].strip())] = int(line.split()[0].strip())

if not os.path.isdir(UCF20Path):
    os.makedirs(UCF20Path)

    for className in classNames.keys():
        shutil.copytree(os.path.join(UCF101Path,className),os.path.join(UCF20Path,className))
        print 'Dir %s done\n'%className