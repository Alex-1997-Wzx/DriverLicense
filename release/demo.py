from __future__ import division
import torch
from torch.autograd import Variable
# import utils
from util.recog_utils import strLabelConverter
# import dataset
from PIL import Image
import torchvision.transforms as transforms

# import models.crnn as crnn
import models.xscrnn as xscrnn

# model_path = './data/crnn.pth'
# img_path = './data/demo.png'
model_path = './weights/cl253_recog_20191129_48_6000.pth'
img_path = r'D:\xiashu\cl253\CL253_DriveringLicense\data\cut_cl20\宾凡龙'

alphabet = ''
with open('./weights/alpha.txt', 'r', encoding='utf-8') as f:
    alphabet = f.readline()

model = xscrnn.XSCRNN(32, 1, len(alphabet)+1, 256)
if torch.cuda.is_available():
    model = model.cuda()

model = torch.nn.DataParallel(model)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

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


import os
import cv2
import numpy as np
total = 0
acc = 0
for f in os.listdir(img_path):
    label = f[:-4].split('_')[-1]
    label = label.lower()

    fn = os.path.join(img_path, f)

    converter = strLabelConverter(alphabet)
    transformer = resizeNormalize((400, 32))
    image = Image.open(fn).convert('L')
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
    print('')
    # print('truth: ', label)
    # print len(label)
    # print(sim_pred)
    # print len(sim_pred)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    cv_img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
    # cv_img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('img', cv_img)
    cv2.waitKey(0)

    total += 1
    if label == sim_pred:
        acc += 1


print('\n')
# print acc
# print total
print('acc = ', acc/total) 
