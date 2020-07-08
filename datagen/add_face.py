''' 在图片中加入人脸 '''
import os
import cv2


class AddFace(object):
    w, h = 1133, 794
    tw, th = 274, 384
    x0, y0 = 820, 380
    x1, y1 = x0+tw, y0+th

    def __init__(self, face_image_dir=''):
        if not face_image_dir:
            self.face_image_dir = r'E:\datasets\Face\19_Multi-Task Facial Landmark (MTFL) dataset\lfw_5590'
        else:
            self.face_image_dir = face_image_dir
        self.faces = os.listdir(self.face_image_dir)
        self.length = len(self.faces)
        self.index = 0


    def add_face_to_img(self, img):
        img_copy = img.copy()
        f = self.faces[self.index]
        fn = os.path.join(self.face_image_dir, f)
        face_image = cv2.imread(fn)
        face_image = cv2.resize(face_image, (self.tw, self.th))
        img_copy[self.y0:self.y1, self.x0:self.x1, : ] = face_image
        if self.index >= self.length-1:
            self.index = 0
        else:
            self.index += 1
        return img_copy