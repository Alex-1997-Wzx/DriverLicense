import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile


AngleModelPb = r'D:\xiashu\OCR\chineseocr\Angle-model.pb'

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
    ROTATE = [0,90,180,270]
    if adjust:
       thesh = 0.05
       xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
       img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
    img = cv2.resize(img,(224,224))
    img = img[..., ::-1].astype(np.float32)
        
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img          = np.array([img])
    
    out = sess.run(predictions, feed_dict={inputImg: img,
                                              keep_prob: 0
                                             })
    
    index = np.argmax(out,axis=1)[0]
    return ROTATE[index]


if __name__ == '__main__':
    img_path = r'D:\xiashu\cl253\CL253_DriveringLicense\data\original\guangxi\guangxi-20191206'
    # img_path = './data/cl_20'
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    for f in os.listdir(img_path):
        fn = os.path.join(img_path, f)
        print(fn)
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
        angle = angle_detect_tf(img)
        print(angle)
        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imencode('.jpg', img, encode_param)[1].tofile(fn)
        cv2.imshow('src', cv2.resize(img, (600, 400)))
        cv2.waitKey(0)