'''
用于检测文本方向：0/90/180/270
'''
# from config import yoloCfg,yoloWeights,opencvFlag
# from config import AngleModelPb,AngleModelPbtxt
# from config import IMGSIZE
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile


IMGSIZE = (608,608)
AngleModelPb = './models/angle_model.pb'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
with gfile.FastGFile(AngleModelPb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
inputImg =  sess.graph.get_tensor_by_name('input_1:0')
predictions = sess.graph.get_tensor_by_name('predictions/Softmax:0')
keep_prob = tf.placeholder(tf.float32)


def angle_detect_tf(img,adjust=True):
    h,w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
       thesh = 0.05
       xmin, ymin, xmax, ymax = int(thesh*w), int(thesh*h), w-int(thesh*w), h-int(thesh*h)
       img = img[ymin:ymax, xmin:xmax]  # 剪切图片边缘
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)

    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img          = np.array([img])
    
    out = sess.run(predictions, feed_dict={inputImg: img, keep_prob: 0})

    index = np.argmax(out, axis=1)[0]
    return ROTATE[index]


if __name__ == '__main__':
    import os
    import time
    for f in os.listdir('./data/cl_image_20'):
        fn = os.path.join('./data/cl_image_20', f)
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
        print('\nstart predict')
        print(f)
        s = time.time()
        res = angle_detect_tf(img, adjust=False)
        e = time.time()
        print('angle:', res)
        print('time cost: ', e-s)
        # r2 = text_detect(img)
        # points = r2[0]

        # print(points)
        # for p in points:
        #     cv2.rectangle(img, (p[0],p[1]), (p[2],p[3]), (0,0,255))
        img = cv2.resize(img, (800, 600))
        cv2.imshow('re', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        # break