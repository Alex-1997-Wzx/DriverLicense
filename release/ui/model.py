import os
import sys
import time
import random
import argparse
import collections
import cv2
import numpy as np
import pyclipper
import Polygon as plg
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn


import util
import util.io as io
import models
from util.pyxsse import xsse as pyxsse


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
parser.add_argument('--resume', nargs='?', type=str, default='./weights/xiashu_detect.pth.tar',
                    help='Path to previous saved model to restart from')
parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                    help='Path to previous saved model to restart from')
parser.add_argument('--kernel_num', nargs='?', type=int, default=3,
                    help='Path to previous saved model to restart from')
parser.add_argument('--scale', nargs='?', type=int, default=4,
                    help='Path to previous saved model to restart from')
parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                    help='Path to previous saved model to restart from')
parser.add_argument('--min_kernel_area', nargs='?', type=float, default=10.0,
                    help='min kernel area')
parser.add_argument('--min_area', nargs='?', type=float, default=500.0,
                    help='min area')
parser.add_argument('--min_score', nargs='?', type=float, default=0.91,
                    help='min score')
# parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
#                     help='min area')
# parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
#                     help='min score')

args = parser.parse_args()


def load_detect_model(filename, pse_scale=args.scale):
    model = models.resnet50(pretrained=True, num_classes=7, scale=pse_scale)
    for param in model.parameters():
        param.requires_grad = False

    print("Loading model and optimizer from checkpoint '{}'".format(filename))
    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))

    # model.load_state_dict(checkpoint['state_dict'])
    d = collections.OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        d[tmp] = value
    model.load_state_dict(d)

    print("Loaded checkpoint '{}' (epoch {})"
            .format(filename, checkpoint['epoch']))
    sys.stdout.flush()

    model.eval()
    return model


def detect_check_angle(image, model, long_size=1280, use_cuda=True, pse_scale=args.scale):
    org_img = image.copy()
    img = org_img[:, :, [2, 1, 0]]

    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    scaled_img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    
    scaled_img = scaled_img.unsqueeze(0)
    if use_cuda:
        with torch.no_grad():
            img = Variable(scaled_img.cuda())
        torch.cuda.synchronize()
    else:
        with torch.no_grad():
            img = Variable(scaled_img)

    # start = time.time()

    outputs = model(img)
    # end = time.time()
    # print('time cost : ', (end-start))

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    
    pred = pyxsse(kernels, args.min_kernel_area / (pse_scale * pse_scale))

    
    label = pred
    label_num = np.max(label) + 1

    x_ratio = org_img.shape[0] / text.shape[0]
    y_ratio = org_img.shape[1] / text.shape[1]

    left_boxes = []  # 左高右低
    right_boxes = []  # 左低右高
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (pse_scale * pse_scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue

        rect = cv2.minAreaRect(points)
        center_x = int(rect[0][0] * x_ratio)
        center_y = int(rect[0][1] * y_ratio)
        width = int(rect[1][0] * x_ratio)
        height = int(rect[1][1] * y_ratio)
        rect_scale = [center_x, center_y, width, height, rect[2]]
        if -90 < rect[2] < -45:
            left_boxes.append(rect_scale)
        else:
            right_boxes.append(rect_scale)

    angle_justify = 0
    if len(left_boxes) > len(right_boxes):
        rects = sorted(left_boxes, key=lambda x: x[3], reverse=True)
        angle_justify += 90
    else:
        rects = sorted(right_boxes, key=lambda x: x[2], reverse=True)
    widthes = []
    for rect in rects[:3]:
        widthes.append(rect[4]+angle_justify)
    if not len(widthes):
        angle = 0
    else:
        angle = sum(widthes) / len(widthes)
    # print('angle: ', angle)

    rows, cols = org_img.shape[0], org_img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rot = cv2.warpAffine(org_img, M, (cols, rows))
    # img_rot = cv2.resize(img_rot, (800, 600))
    # cv2.imshow('rot', img_rot)
    # cv2.waitKey(0)

    # 寻找证件位置
    # 1.计算旋转后的坐标
    new_rects = []
    left_boxes.extend(right_boxes)
    for rect in left_boxes:
        tp_rect = ((rect[0],rect[1]), (rect[2],rect[3]), rect[4])
        points = cv2.boxPoints(tp_rect)

        old_points = np.array(points, dtype=np.int32)
        old_points = np.reshape(old_points, (4,1,2))
        new_points = cv2.transform(old_points, M)

        new_xy = []
        for i in range(4):
            new_xy.append((new_points[i][0][0], new_points[i][0][1]))
        new_rects.append(cv2.boundingRect(np.array(new_xy)))

    x,y,w,h = caculate_area(img_rot, new_rects)
    # cv2.rectangle(img_rot,(x,y),(x+w,y+h),(0,0,255),3)
    # for rect in new_rects:
    #     cv2.rectangle(img_rot, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255), 2)
    # img_resize = cv2.resize(img_rot, (800, 600))
    # cv2.imshow('rot', img_resize)
    # cv2.waitKey(0)

    # return img_rot[y:y+h, x:x+w, :], angle
    return img_rot[y:y+h, x:x+w, :], round(angle, 2), (x, y, w, h)


def caculate_area(img, rects):
    h_, w_ = img.shape[:2]
    # 求文本平均高度
    all_height = [x[3] for x in rects]
    height_avg = int(sum(all_height) / len(all_height))
    # print('height_avg: ', height_avg)
    half_height = height_avg // 2

    # 求文本中心点
    all_x = [x[0]+x[2]//2 for x in rects]
    x_avg = int(sum(all_x) / len(all_x))
    all_y = [x[1]+x[3]//2 for x in rects]
    y_avg = int(sum(all_y) / len(all_y))

    # template image
    empty = np.zeros((h_, w_), np.uint8)

    # 首先中间填个白色区域（目的是为了联通上下两个区域，驾驶证往往中间会隔开）
    bias = 4 * height_avg  # 沿中心扩大文本高度的4倍
    empty[y_avg-bias: y_avg+bias, x_avg-bias: x_avg+bias] = 255

    for rect in rects:
        row_min = max(0, rect[1]-height_avg)
        row_max = min(rect[1]+rect[3]+height_avg, h_)
        col_min = max(rect[0]-height_avg, 0)
        col_max = min(rect[0]+rect[2]+height_avg, w_)

        empty[row_min:row_max, col_min:col_max] = 255
    th = int(15 * h_ / height_avg)
    tw = int(th * w_ / h_)
    empty = cv2.resize(empty, (tw, th))
    empty = cv2.dilate(empty, (7, 7), iterations=2)
    # cv2.imshow('bin', cv2.resize(empty, (800, 600)))
    contours, _ = cv2.findContours(empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    x,y,w,h = cv2.boundingRect(cnt)

    x = int(x * w_ / tw)
    y = int(y * h_ / th)
    w = int(w * w_ / tw)
    h = int(h * h_ / th)
    # tempimg = cv2.resize(img, (tw, th))
    # print((x, y, w, h))
    # cv2.rectangle(tempimg,(x,y),(x+w,y+h),(0,0,255),3)
    
    # cv2.imshow('bin', tempimg)
    # cv2.waitKey(0)
    return x, y, w, h


def detect_boxes(image, model, long_size=1280, use_cuda=True, pse_scale=args.scale):
    org_img = image.copy()
    # org_img, angle = detect_check_angle(image, model)
    img = org_img[:, :, [2, 1, 0]]

    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    scaled_img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    
    scaled_img = scaled_img.unsqueeze(0)
    if use_cuda:
        with torch.no_grad():
            img = Variable(scaled_img.cuda())
        torch.cuda.synchronize()
    else:
        with torch.no_grad():
            img = Variable(scaled_img)

    # start = time.time()

    outputs = model(img)
    # end = time.time()
    # print('time cost : ', (end-start))

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    
    pred = pyxsse(kernels, args.min_kernel_area / (pse_scale * pse_scale))

    
    label = pred
    label_num = np.max(label) + 1

    x_ratio = org_img.shape[0] / text.shape[0]
    y_ratio = org_img.shape[1] / text.shape[1]
    # print('x_scaled: ', x_ratio)
    # print('y_scaled: ', y_ratio)
    bboxes = []
    heights = []  # zp add: 20191206
    widths = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (pse_scale * pse_scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue

        # rect_ = cv2.minAreaRect(points)
        # rect = []
        # rect.append((rect_[0][0] * x_ratio, rect_[0][1] * y_ratio))
        # rect.append((rect_[1][0] * x_ratio, rect_[1][1] * y_ratio))
        # rect.append(rect_[2])
        # angle = rect[2]
        # print(angle)
        # if -90 < angle < -45:
        #     angle += 90
        # rows, cols = org_img.shape[0], org_img.shape[1]
        # if rect[1][0] > 20:
        #     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        #     img_rot = cv2.warpAffine(org_img,M,(cols,rows))
        #     print(rect)

        #     # rotate bounding box
        #     rect0 = tuple(rect)
        #     box = cv2.boxPoints(rect0)
        #     pts = np.int0(cv2.transform(np.array([box]), M))[0]    
        #     pts[pts < 0] = 0
        #     # bbox = (pts[1][0], pts[1][1], pts[2][0] - pts[1][0], pts[0][1] - pts[1][1])
        #     # crop
            
        #     h_ = abs(pts[0][1] - pts[1][1])
        #     h_bias = int(h_ * 0.2)
        #     l_bias = int(h_ * 0.05)

        #     img_crop = img_rot[pts[1][1] + l_bias : pts[0][1] + h_bias, 
        #                     pts[1][0] - l_bias : pts[2][0] + l_bias]
        #     cv2.imwrite('img/img_'+str(i)+'.jpg', img_crop)

        rect = cv2.minAreaRect(points)
        # print('angle ', rect[2])


        bbox = cv2.boundingRect(points)
        if bbox is not None:
            x0 = int(bbox[0] * x_ratio)
            y0 = int(bbox[1] * y_ratio)
            x1 = x0 + int(bbox[2] * x_ratio)
            y1 = y0 + int(bbox[3] * y_ratio)

            # y1 = min(org_img.shape[1], y1+5)

            # x0, y0, x1, y1 = x0 - 5, y0 - 5, x1 + 5, y1 + 5
            # x0 -= ps_bias_1
            # y0 -= ps_bias_1
            # x1 += ps_bias_2
            # y1 += ps_bias_2
            x0 = max(0, x0-3)
            y0 = max(0, y0+2)
            x1 = min(x1+3, (org_img.shape[1] - 1))
            y1 = min(y1+3, (org_img.shape[0] - 1))

            # cut_img = org_img[y0:y1, x0:x1, :]
            # h, w = cut_img.shape[:2]
            # if -90 < rect[2] < -45:
            #     angle = 90 + rect[2]
            # else:
            #     angle = rect[2]
            # M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            # cut_img = cv2.warpAffine(cut_img, M, (w, h))
            # cv2.imshow('cut', cut_img)
            # cv2.waitKey(0)

            bboxes.append((x0, y0, x1, y1, rect[2]))
            heights.append(y1-y0)
            widths.append(x1-x0)

        # rect = cv2.minAreaRect(points)
        # bbox = cv2.boxPoints(rect)
        # if bbox.shape[0] <= 2:
                # continue
        # bbox = bbox * (x_ratio, y_ratio)
        # bbox = bbox.astype('int32')
        # bboxes.append(bbox.reshape(-1))

    # re order
    avg_h = int(sum(heights) / len(heights))
    avg_w = int(sum(widths) / len(widths))
    thresh_h = int(avg_h * 0.6)

    re_boxes = []
    items = []
    avg_y = 0
    for box in bboxes:
        if not items:  # first item
            avg_y = box[1]
            items.append(box)
            continue
        # caculate height difference
        if abs(box[1] - avg_y) < thresh_h:
            avg_y = (avg_y * len(items) + box[1]) / ((len(items) + 1))
            items.append(box)
        else:  # new line
            if len(items):
                items.sort(key=lambda x: x[0])
                re_boxes.extend(items)
                items.clear()
                avg_y = 0
            avg_y = box[1]
            items.append(box)
    if len(items):
        items.sort(key=lambda x: x[0])
        re_boxes.extend(items)

    if use_cuda:
        torch.cuda.synchronize()

    # sys.stdout.flush()

    # for bbox in bboxes:
    #     x0, y0, x1, y1 = bbox
    #     cv2.rectangle(org_img, (x0, y0), (x1, y1), (0, 0, 255), 3)
        # cv2.drawContours(org_img, [bbox.reshape(bbox.shape[0] // 2, 2)], -1, (0, 0, 255), 2)
    # cv2.imwrite('./img/00000.jpg', org_img)

    return re_boxes, avg_w, avg_h


def count_boxes(image, model, long_size=1280, use_cuda=True, pse_scale=args.scale):
    org_img = image.copy()
    img = org_img[:, :, [2, 1, 0]]

    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    scaled_img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    scaled_h, scaled_w = scaled_img.shape[:2]
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    
    scaled_img = scaled_img.unsqueeze(0)
    if use_cuda:
        with torch.no_grad():
            img = Variable(scaled_img.cuda())
        torch.cuda.synchronize()
    else:
        with torch.no_grad():
            img = Variable(scaled_img)

    # start = time.time()

    outputs = model(img)
    # end = time.time()
    # print('time cost : ', (end-start))

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    
    pred = pyxsse(kernels, args.min_kernel_area / (pse_scale * pse_scale))

    
    label = pred
    label_num = np.max(label) + 1

    count = 0
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (pse_scale * pse_scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue
        bbox = cv2.boundingRect(points)
        if abs(bbox[2]-bbox[0]) / scaled_w > 0.2:
            count += 1

    return count


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


from util.recog_utils import strLabelConverter
import models.xscrnn as xscrnn

alphabet = ''
with open('./weights/alpha.txt', 'r', encoding='utf-8') as f:
    alphabet = f.readline()

def load_recognize_model(filename):
    model = xscrnn.XSCRNN(32, 1, len(alphabet)+1, 256)
    print('loading pretrained model from %s' % filename)
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(filename))
    else:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return model


converter = strLabelConverter(alphabet)
transformer = resizeNormalize((400, 32))

def recognize_one(image, model, use_cuda=True):
    image = Image.fromarray(image).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    raw_pred = raw_pred.upper()
    sim_pred = sim_pred.upper()
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred


def load_angle_model(path):
    vgg16 = vgg16_bn(pretrained=False, num_classes=4)
    for param in vgg16.features.parameters():
        param.require_grad = False
    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 4)]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    if torch.cuda.is_available():
        vgg16.cuda()
        vgg16 = torch.nn.DataParallel(vgg16)
    vgg16.load_state_dict(torch.load(path))
    return vgg16

def angle_rectify(anglemodel, image_input, use_gpu=True):
    angles = [0, 180, 270, 90]
    image = image_input[:, :, [2, 1, 0]]
    scaled_img = cv2.resize(image, (256, 256))

    h, w = scaled_img.shape[0:2]
    th, tw = (224, 224)
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    scaled_img = scaled_img[i:i+th, j:j+tw, :]
    scaled_img = transforms.ToTensor()(scaled_img)

    scaled_img = scaled_img.unsqueeze(0)
    if use_gpu:
        img = Variable(scaled_img.cuda(), volatile=True)
        torch.cuda.synchronize()
    else:
        img = Variable(scaled_img, volatile=True)

    outputs = anglemodel(img)
    print(outputs)
    index = int(outputs.argmax())
    return angles[index]


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    model_detect = load_detect_model(args.resume)
    model_path = './weights/xiashu_recognize.pth'
    model_recognize = load_recognize_model(model_path)

    root_dir = './data/test/'
    for f in os.listdir(root_dir):
        fn = os.path.join(root_dir, f)
        image = cv2.imread(fn)
        boxes = detect_boxes(image, model_detect, use_cuda=use_cuda)

        for box in boxes:
            img = image[box[1]:box[3], box[0]:box[2]]
            text = recognize_one(img, model_recognize)
            fontScale = (box[3]-box[1]) / 64
            cv2.rectangle(image, box[:2], box[2:], (0, 0, 255), 2)
            cv2.putText(image, text, box[:2], cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)

        cv2.imwrite('outputs/vis_ctw1500/' + f, image)

