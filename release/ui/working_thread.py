
import os
import time
import math
import base64
import cv2
from PyQt5 import QtCore
import numpy as np

import ui.config as cfg
from ui.model import load_detect_model, load_recognize_model
from ui.model import recognize_one, detect_boxes, detect_check_angle
from ui.model import load_angle_model, angle_rectify
from ui.utils import contrast_one
import torch

from util.post_processing import text_only_information_handle


class TestThread(QtCore.QThread):
    
    testFinishedSignal = QtCore.pyqtSignal(name='testFinishedSignal')
    progressSignal = QtCore.pyqtSignal(int, name='progressSignal')
    errorMsgSignal = QtCore.pyqtSignal(str, name='errorMsgSignal')
    msgSignal = QtCore.pyqtSignal(str, name='msgSignal')
    massageBoxSignal = QtCore.pyqtSignal(str, name='massageBoxSignal')
    modelLoaded = QtCore.pyqtSignal()

    def __init__(self, angle_md=None, detect_md=None, recog_md=None, lexicon_file=None):
        super().__init__()
        self.testfolder = None

        self.angle_md = angle_md
        self.detect_md = detect_md
        self.recog_md = recog_md
        self.lexicon_file = lexicon_file

        self.loadflag = False  # 加载模型标志
        self.results = []  # 测试结果
        self.currentResult = None
        self.exitThread = False
        self.wake = False
        self.mutex = QtCore.QMutex()
        self.taskAdded = QtCore.QWaitCondition()

        self.tempCount = 0

    def exit_thread(self):
        self.exitThread = True
        self.wake = False
        self.taskAdded.wakeOne()

    def load_model(self, angle_md=None, detect_md=None, recog_md=None, lexicon_file=None):
        self.angle_md = angle_md
        self.detect_md = detect_md
        self.recog_md = recog_md
        self.lexicon_file = lexicon_file
        with QtCore.QMutexLocker(self.mutex):
            self.loadflag = True
            self.exitThread = False
            self.wake = True
            self.taskAdded.wakeOne()

    def add_task(self, testfolder):
        with QtCore.QMutexLocker(self.mutex):
            self.testfolder = testfolder
            self.exitThread = False
            self.wake = True
            self.taskAdded.wakeOne()

    def run(self):
        model_angle = None
        model_detect = None
        model_recognize = None
        use_cuda = torch.cuda.is_available()
        while True:
            with QtCore.QMutexLocker(self.mutex):
                if not self.wake:
                    self.taskAdded.wait(self.mutex)
                if self.exitThread:
                    break
            self.wake = False

            if self.loadflag:  # load model
                self.loadflag = False
                # try:
                # self.msgSignal.emit('ready for load model.')
                # print('prepare load model')
                model_angle = load_angle_model(self.angle_md) if self.angle_md else None
                # print(self.detect_md)
                model_detect = load_detect_model(self.detect_md, pse_scale=4) if self.detect_md else None
                # print(self.recog_md)
                model_recognize = load_recognize_model(self.recog_md) if self.recog_md else None
                # print('load model success.')
                self.modelLoaded.emit()
                self.msgSignal.emit('success load model.')
                # except Exception as e:
                    # self.errorMsgSignal.emit(str(e))
                    # break
                continue

            self.results.clear()

            if not model_detect or not model_recognize:
                self.errorMsgSignal.emit('please load detect and recognize model first!')
                self.testFinishedSignal.emit()
                continue

            if not self.testfolder or not os.path.exists(self.testfolder):
                self.errorMsgSignal.emit('test folder error: ' + str(self.testfolder))
                self.testFinishedSignal.emit()
                continue

            allfiles = [f for f in os.listdir(self.testfolder) if os.path.splitext(f)[-1].lower() in ['.jpg', '.png', '.jpeg']]
            length = len(allfiles)
            for i, f in enumerate(allfiles):
                start = time.time()
                if os.path.splitext(f)[-1].lower() not in ['.jpg', '.png', '.jpeg']:
                    continue
                fn = os.path.join(self.testfolder, f)
                print(fn)
                image_org = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = image_org.copy()
                
                # image = contrast_one(image)

                if model_angle is not None:
                    angle = angle_rectify(model_angle, image, use_cuda)
                    if angle == 270:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle == 180:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    elif angle == 90:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                # 求微调角度和证件文本位置框
                rot_image, angle_adjust, rect = detect_check_angle(image, model_detect, use_cuda=use_cuda, long_size=1080, pse_scale=8)
                
                # cv2.imwrite('./img/{}.jpg'.format(self.tempCount), rot_image)
                self.tempCount += 1

                boxes, avg_w, avg_h = detect_boxes(rot_image, model_detect, use_cuda=use_cuda, long_size=1280, pse_scale=4)

                item = {}
                item['filename'] = fn
                # item['detect_angle'] = str(round(detect_angle, 2))
                
                if self.angle_md:
                    item['angle'] = str(angle)
                item['angle_adjust'] = str(angle_adjust)  # 微调角度
                item['rect'] = list(rect)  # 证件框
                item['width'] = image_org.shape[1]
                item['height'] = image_org.shape[0]
                item['imagedata'] = base64.b64encode(image).decode('utf-8') if cfg.ImageData else ''
                item['labels'] = []
                for box in boxes:
                    h, w = rot_image.shape[:2]
                    if min(box[:4]) < 0:
                        print('found box number < 0')
                        continue
                    if max(box[1], box[3]) >= h:
                        print('found box h >= ', h)
                        continue
                    if max(box[0], box[2]) >= w:
                        print('found box w >=', w)
                        continue
                    if box[1] >= box[3] or box[0] >= box[2]:
                        print(box)
                        continue

                    img = rot_image[box[1]:box[3], box[0]:box[2]]
                    if (box[2]-box[0]) > avg_w:
                        if -90 < box[4] < -45:
                            agl = 90 + box[4]
                        else:
                            agl = box[4]
                        if abs(agl) > 1:
                            h, w = img.shape[:2]
                            M = cv2.getRotationMatrix2D((w/2, h/2), agl, 1)
                            img = cv2.warpAffine(img, M, (w, h))
                            bias_h = int(w * math.tan(agl * math.pi / 180) / 2) - 2
                            bias_h = max(0, bias_h)
                            img = img[bias_h:h-bias_h, :, :]

                    # import random
                    # fname = './img/cut/' + str(random.randint(10000000, 1000000000)) + '.jpg'
                    # cv2.imwrite(fname, img)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)
                    text = recognize_one(img, model_recognize)
                    # fontScale = (box[3]-box[1]) / 64
                    # cv2.rectangle(image, box[:2], box[2:], (0, 0, 255), 2)
                    # cv2.putText(image, text, box[:2], cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
                    box_item = {}
                    box_item['box'] = box[:4]
                    box_item['label'] = text
                    item['labels'].append(box_item)
                # cv2.imwrite('outputs/vis_ctw1500/' + f, image)
                # save informations
                texts = []
                boxes_ = []
                for label in item['labels']:
                    texts.append(label['label'])
                    boxes_.append(label['box'][:4])
                flag, re_dict = text_only_information_handle(texts, boxes_)
                if not flag:
                    print('recognize fail.')
                    continue
                item['label_show'] = list(re_dict.values())

                self.results.append(item)
                self.progressSignal.emit(int(100 * (i+1) / length))

                end = time.time()
                # print('all time: ', end-start)

                # cv2.imwrite('./regular/origin/' + str(self.tempCount) + '.jpg', rot_image)
                # for label in item['labels']:
                #     texts.append(label['label'] + ' '+','.join([str(x) for x in label['box']]))
                # with open('./regular/origin/' + str(self.tempCount) + '.txt', 'w', encoding='utf-8') as f:
                #     f.write('\n'.join(texts))

                # cv2.imwrite('./regular/target/' + str(self.tempCount) + '.jpg', rot_image)
                # with open('./regular/target/' + str(self.tempCount) + '.txt', 'w', encoding='utf-8') as f:
                #     entries = []
                #     for key, val in re_dict.items():
                #         entries.append(key+':'+val)
                #     f.write('\n'.join(entries))

                # lot_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
                
                entries = []
                entries.append('证号:' + re_dict['driver_id'])
                entries.append('有效期限:' + re_dict['end'])
                entries.append('准驾车型:' + re_dict['class'])
                entries.append('住址:' + re_dict['address'])
                entries.append('至:' + re_dict['end'])
                entries.append('姓名:' + re_dict['name'])
                entries.append('国籍:' + re_dict['nation'])
                entries.append('出生日期:' + re_dict['brith'])
                entries.append('性别:' + re_dict['gender'])
                entries.append('初次领证日期:' + re_dict['first'])
                
                save_file = os.path.join('./data/target', f.split('.')[0]+'.txt')
                with open(save_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(entries))

            self.testFinishedSignal.emit()