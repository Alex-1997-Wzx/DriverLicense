''' 将高清图片转为低清图片，节省数据空间，驾驶证本身较模糊，也没必要高清 '''
import os
import cv2
import numpy as np

root_dir = './material/gaoqing'
output = './material/diqing'


encode = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
for f in os.listdir(root_dir):
    if f[-4:] != '.jpg':
        continue
    fn = os.path.join(root_dir, f)
    img = cv2.imread(fn)
    gtfn = os.path.join(output, f)
    cv2.imencode('.jpg', img, encode)[1].tofile(gtfn)


