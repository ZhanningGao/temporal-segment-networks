# Since most of the test and val videos have no gt labels, we first extract those videos with gt to a new folder

import os
import shutil
import scipy.io as sio

gtpath = '../data/TH14evalkit/groundtruth'

ValPath = '../data/TH14evalkit/groundtruth'
TestPath = '../data/TH14_Temporal_Annotations_Test/annotations/annotation'

TH14path = '/data/datasets/THUMOS14'

detClassFile = open(os.path.join(gtpath, 'detclasslist.txt'))

classNames = dict()

for line in detClassFile.readlines():
    classNames[(line.split()[-1].strip())] = int(line.split()[0].strip())

videoNames = set()
for className in classNames.keys():
    valfilename = className + '_' + 'val' + '.txt'
    try:
        valfile = open(os.path.join(ValPath, valfilename))
    except IOError:
        print 'ERROR: Cannot open the gt file!'

    for videoFile in valfile.readlines():
        videoName = videoFile.split()[0].strip()
        # Dict for gt event info
        videoNames.add(videoName)
videoList = []
for videoName in videoNames:
    videoList.append(videoName)
    if not os.path.isfile(os.path.join(TH14path, 'TRUEval', videoName + '.mp4')):
        print 'Copying %s ...'%videoName
        shutil.copyfile(os.path.join(TH14path,'val',videoName+'.mp4'),os.path.join(TH14path,'TRUEval',videoName+'.mp4'))
        print ' done\n'

sio.savemat('../data/TRUEval_set_name.mat', {'valVideoNames': videoList})

videoNames = set()
for className in classNames.keys():
    testfilename = className + '_' + 'test' + '.txt'
    try:
        testfile = open(os.path.join(TestPath, testfilename))
    except IOError:
        print 'ERROR: Cannot open the gt file!'

    for videoFile in testfile.readlines():
        videoName = videoFile.split()[0].strip()
        # Dict for gt event info
        videoNames.add(videoName)
videoList = []
for videoName in videoNames:
    videoList.append(videoName)
    if not os.path.isfile(os.path.join(TH14path, 'TRUEtest', videoName + '.mp4')):
        print 'Copying %s ...'%videoName
        shutil.copyfile(os.path.join(TH14path,'test',videoName+'.mp4'),os.path.join(TH14path,'TRUEtest',videoName+'.mp4'))
        print ' done\n'

sio.savemat('../data/TRUEtest_set_name.mat', {'testVideoNames': videoList})

