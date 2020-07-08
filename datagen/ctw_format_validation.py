''' 可视化验证label_to_ctw_format.py转换后的标注数据是否准确。
'''
import os
import cv2
import numpy as np

# root_label_dir = './gen/labels'
ctw_label_dir = './gen/labels_ctw'
root_image_dir = './gen/original'

ctw_label_dir = './data/labeled/train/text_label_curve'
root_image_dir = './data/labeled/train/text_image'

for f in os.listdir(root_image_dir):
    fn = os.path.join(root_image_dir, f)
    img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
    txtfile = os.path.join(ctw_label_dir, f[:-4]+'.txt')
    with open(txtfile, 'r', encoding='utf-8') as tf:
        lines = tf.readlines()
        for line in lines:
            lb = line.split(',')
            lb = [int(k) for k in lb]
            x0,y0,x2,y2 = lb[:4]
            cv2.rectangle(img, (x0,y0), (x2,y2), (0,0,255), 2)
            for i in range(len(lb[4:])//2):
                bias_x = lb[4:][2*i]
                bias_y = lb[4:][2*i+1]
                cx = x0 + bias_x
                cy = y0 + bias_y
                cv2.circle(img, (cx, cy), 3, (0,0,255), 2)
    img = cv2.resize(img, (800, 600))
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
