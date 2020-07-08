import os
import cv2
import numpy as np

from ui.model import load_detect_model, detect_boxes, detect_check_angle


model = load_detect_model('./weights/checkpoint_fine_tuning.pth.tar')

imagepath = r'D:\xiashu\cl253\CL253_DriveringLicense\data\original\guangxi\guangxi-20191206'
for f in os.listdir(imagepath):
    # if 'fdvfds' not in f:
    #     continue
    fname = os.path.join(imagepath, f)

    # img = cv2.imread(imagepath)
    print(f)
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
    # print(img.shape)
    # detect_boxes(img, model)
    img_rot, angle, rect = detect_check_angle(img, model)
    # boxes, avg_w, avg_h = detect_boxes(img_rot, model, use_cuda=True, long_size=1280, pse_scale=4)
    img_rot = cv2.resize(img_rot, (600, 400))
    img = cv2.resize(img, (600, 400))
    cv2.imshow('src', img)
    cv2.imshow('sdf', img_rot)
    key = cv2.waitKey(0)

    # for box in boxes:
    #     h, w = img_rot.shape[:2]
    #     if min(box[:4]) < 0:
    #         print('found box number < 0')
    #         continue
    #     if max(box[1], box[3]) >= h:
    #         print('found box h >= ', h)
    #         continue
    #     if max(box[0], box[2]) >= w:
    #         print('found box w >=', w)
    #         continue
    #     if box[1] >= box[3] or box[0] >= box[2]:
    #         print(box)
    #         continue

    #     img = img_rot[box[1]:box[3], box[0]:box[2]]
    #     if (box[2]-box[0]) > avg_w:
    #         if -90 < box[4] < -45:
    #             agl = 90 + box[4]
    #         else:
    #             agl = box[4]
    #         if abs(agl) > 1:
    #             h, w = img.shape[:2]
    #             M = cv2.getRotationMatrix2D((w/2, h/2), agl, 1)
    #             img = cv2.warpAffine(img, M, (w, h))

    #             import math
    #             bias_h = int(w * math.tan(agl * math.pi / 180) / 2)
    #             img = img[bias_h:h-bias_h, :, :]
    #             cv2.imshow('img', img)
    #             cv2.waitKey(0)
    
    if key == ord('q'):
        break