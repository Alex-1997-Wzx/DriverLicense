import cv2
import numpy as np
import os

imagepath = './data/bowen'
for f in os.listdir(imagepath):
    fname = os.path.join(imagepath, f)
    image = cv2.imread(fname)
    # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpen = cv2.filter2D(image, -1, sharpen_kernel)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    target = cv2.merge((bH, gH, rH))

    # target = cv2.equalizeHist(gray)


    cv2.imshow('origin', cv2.resize(image, (600, 400)))
    cv2.imshow('target', cv2.resize(target, (600, 400)))
    cv2.waitKey(0)