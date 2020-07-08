#-*- coding:utf-8 -*-
''' 将图片、标注打包到一个数据库中 '''
import os
import lmdb
import cv2
import glob
import numpy as np
 
 
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True
 
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
 
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
#    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # print (len(imagePathList) , len(labelList))
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print '...................'
    env = lmdb.open(outputPath, map_size=8589934592)#1099511627776)所需要的磁盘空间的最小值，之前是1T，我改成了8g，否则会报磁盘空间不足，这个数字是字节
    
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)#注意一定要在linux下，否则f.read就不可用了，就会输出这个信息
                continue
 
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
 
 
def read_text(path):
    
    with open(path) as f:
        text = f.read()
    text = text.strip()
    
    return text
 
 
if __name__ == '__main__':
    # lmdb 输出目录
    outputPath = 'D:/ruanjianxiazai/tuxiangyangben/fengehou/train'#训练集和验证集要跑两遍这个程序，分两次生成
 
    path = "D:/ruanjianxiazai/tuxiangyangben/fengehou/chenguang/*.jpg"#将txt与jpg的都放在同一个文件里面
    imagePathList = glob.glob(path)
    print '------------',len(imagePathList),'------------'
    imgLabelLists = []
    for p in imagePathList:
        try:
           imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue
            
    # imgLabelList = [ (p, read_text(p.replace('.jpg', '.txt'))) for p in imagePathList]
    # sort by labelList
    imgLabelList = sorted(imgLabelLists, key = lambda x:len(x[1]))
    imgPaths = [ p[0] for p in imgLabelList]
    txtLists = [ p[1] for p in imgLabelList]
    
    createDataset(outputPath, imgPaths, txtLists, lexiconList=None, checkValid=True)