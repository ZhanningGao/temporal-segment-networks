import os
import numpy as np

def TH14evaldet(detfilename, gtpath, subset):

    # Read ground truth

    detClassFile = open(os.path.join(gtpath, 'detclasslist.txt'))

    classNames = dict()
    classNamesAll = dict()
    for line in detClassFile.readlines():
        classNames[(line.split()[-1].strip())] = int(line.split()[0].strip())
        classNamesAll[(line.split()[-1].strip())] = int(line.split()[0].strip())

    classNamesAll['Ambiguous'] = 102

    indName = dict((value, key) for key, value in classNames.iteritems())

    gtEvents = []
    for className in classNamesAll.keys():
        gtfilename = className + '_' + subset + '.txt'
        try:
            gtfile = open(os.path.join(gtpath, gtfilename))
        except IOError:
            print 'ERROR: Cannot open the gt file!'

        for videoFile in gtfile.readlines():
            gtEvent = dict()
            videoName = videoFile.split()[0].strip()
            startTime = float(videoFile.split()[1].strip())
            endTime   = float(videoFile.split()[2].strip())
            # Dict for gt event info
            gtEvent['videoName'] = videoName
            gtEvent['timeInterval'] = [startTime, endTime]
            gtEvent['className'] = className
            gtEvent['conf'] = 1

            gtEvents.append(gtEvent)

    # Parse detection results

    try:
        detFile = open(os.path.join(gtpath, detfilename))
    except IOError:
        print 'ERROR: Cannot find the dection result file %s\n'%detfilename

    detEvents = []
    for detFileName in detFile.readlines():
        detEvent = dict()
        strList = detFileName.split()
        if indName.get(int(strList[3].strip())):
            detEvent['videoName'] = strList[0].split('.')[0].strip()
            detEvent['timeInterval'] = [float(strList[1].strip()), float(strList[2].strip())]
            detEvent['className'] = indName[int(strList[3].strip())]
            detEvent['conf'] = float(strList[-1].strip())

            detEvents.append(detEvent)
        else:
            print 'WARNING: Reported class ID %d is not among THUMOS14 detection classes.\n'

    # Evaluate per-class PR for multiple overlap thresholds

    overlapthreshall = [0.1, 0.2, 0.3, 0.4, 0.5]
    ap_all = []
    pr_all = []
    map = []

    for olap in overlapthreshall:
        ap_c = []
        pr_ = dict()
        pr_c = []
        for className in classNames.keys():
            ind = classNames[className]
            rec, prec, ap = TH14evaldetpr(detEvents, gtEvents, className, olap)
            pr_['classInd'] = ind
            pr_['overlapThresh'] = olap
            pr_['prec'] = prec
            pr_['ap'] = ap
            ap_c.append(ap)
            pr_c.append(pr_)

            print 'AP:%1.3f at overlap %1.1f for %s\n'%(ap,olap,className)

        ap_all.append(ap_c)
        pr_all.append(pr_c)
        map.append(sum(ap_c)/len(ap_c))

    return pr_all,ap_all,map

def TH14evaldetpr(detEvents, gtEvents, className, olap):

    videoNames = set()
    detEventSub = []
    gtEventSub = []
    ambEventSub = []
    for detEvent in detEvents:
        videoNames.add(detEvent['videoName'])
        if detEvent['className'] == className:
            detEventSub.append(detEvent)
    for gtEvent in gtEvents:
        videoNames.add(gtEvent['videoName'])
        if gtEvent['className'] == 'Ambiguous':
            ambEventSub.append(gtEvent)
        if gtEvent['className'] == className:
            gtEventSub.append(gtEvent)

    npos = len(gtEventSub)

    assert npos>0

    tpConf = np.array([])
    fpConf = np.array([])

    for videoName in videoNames:
        detEventSubVideo = []
        gtEventSubVideo = []
        ambEventSubVideo = []
        for detEvent in detEventSub:
            if detEvent['videoName'] == videoName:
                detEventSubVideo.append(detEvent)
        for gtEvent in gtEventSub:
            if gtEvent['videoName'] == videoName:
                gtEventSubVideo.append(gtEvent)
        for ambEvent in ambEventSub:
            if ambEvent['videoName'] == videoName:
                ambEventSubVideo.append(ambEvent)

        if len(detEventSubVideo):
            detEventSubVideo.sort(lambda x, y: cmp(x['conf'], y['conf']), reverse=True)
            conf = np.array([detEvent['conf'] for detEvent in detEventSubVideo])
            indFree = np.ones(len(detEventSubVideo))
            indAmb  = np.zeros(len(detEventSubVideo))

            if len(gtEventSubVideo):
                ov = IntervalOverlapSeconds([gtEvent['timeInterval'] for gtEvent in gtEventSubVideo],
                                            [detEvent['timeInterval'] for detEvent in detEventSubVideo])
                for ov_r in ov:
                    if sum(indFree):
                        npov = np.array(ov_r)
                        npov[indFree==0] = 0
                        if npov.max() > olap:
                            indFree[npov.argmax()] = 0
            if len(ambEventSubVideo):
                ovamb = IntervalOverlapSeconds([ambEvent['timeInterval'] for ambEvent in ambEventSubVideo],
                                            [detEvent['timeInterval'] for detEvent in detEventSubVideo])
                indAmb = np.array(ovamb).sum(0)

            fpConf = np.append(fpConf, conf[indFree==1][indAmb[indFree==1]==0])
            tpConf = np.append(tpConf, conf[indFree==0])

    Conf = np.append(tpConf, fpConf)
    ConfInd = np.append(np.ones(tpConf.shape),2*np.ones(fpConf.shape))

    ConfInd = ConfInd[np.argsort(-Conf)]

    TP = np.zeros(Conf.shape)
    FP = np.zeros(Conf.shape)

    TP[ConfInd==1] = 1
    FP[ConfInd==2] = 1

    TP = np.cumsum(TP)
    FP = np.cumsum(FP)

    rec = TP/npos
    prec = TP/(FP+TP)
    ap = prap(rec,prec)

    return rec, prec, ap

def prap(rec,prec):
    ap = 0.0

    recallpoints = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

    for t in recallpoints:
        if len(prec[rec>=t]):
            p = prec[rec>=t].max()
        else:
            p = 0
        ap += p/len(recallpoints)

    return ap

def IntervalOverlapSeconds(i1,i2,normtype=0):
    ov = []

    for ii1 in i1:
        ov_r = []
        for ii2 in i2:
            ov_e = IntervalSingleOverlapSeconds(ii1,ii2,normtype)
            ov_r.append(ov_e)
        ov.append(ov_r)

    return ov

def IntervalSingleOverlapSeconds(ii1,ii2,normtype):
    i1 = np.sort(np.array(ii1))
    i2 = np.sort(np.array(ii2))

    ov = 0.0
    if normtype<0:
        ua = 1.0
    elif normtype == 1:
        ua = i1[1]-i1[0]
    elif normtype == 2:
        ua = i2[1]-i2[0]
    else:
        ua = np.max([i1[1],i2[1]]) - np.min([i1[0],i2[0]])

    iw = np.min([i1[1],i2[1]]) - np.max([i1[0],i2[0]])

    if iw > 0:
        ov = iw/ua

    return ov


#if __name__ == '__main__':

#    pr_all, ap_all, map = TH14evaldet('../results/Run-2-det.txt','../data/TH14evalkit/groundtruth/','val')

#    print map