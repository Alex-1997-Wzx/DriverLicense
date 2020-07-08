from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

class_names = ['0', '180', '270', '90']

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

vgg16 = models.vgg16_bn(pretrained=True)
print(vgg16.classifier[6].out_features) # 1000 
# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)

vgg16.load_state_dict(torch.load('./weights/xiashu_angle.pth'))
if use_gpu:
    vgg16.cuda()

# transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])

org_img = cv2.imread('./data/test/11.jpg')
image = org_img[:, :, [2, 1, 0]]

# scale = long_size * 1.0 / max(h, w)
scaled_img = cv2.resize(image, (256, 256))

h, w = scaled_img.shape[0:2]
th, tw = (224, 224)
i = int(round((h - th) / 2.))
j = int(round((w - tw) / 2.))

scaled_img = scaled_img[i:i+th, j:j+tw, :]

# scaled_img = Image.fromarray(scaled_img)
# scaled_img = scaled_img.convert('RGB')
scaled_img = transforms.ToTensor()(scaled_img)
# scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

scaled_img = scaled_img.unsqueeze(0)
if use_gpu:
    img = Variable(scaled_img.cuda(), volatile=True)
    torch.cuda.synchronize()
else:
    img = Variable(scaled_img, volatile=True)

# start = time.time()

outputs = vgg16(img)
print(type(outputs))
print(outputs)
# print(list(outputs.cpu()))
index = int(outputs.argmax())
print(int(index))


angle = int(class_names[index])

if angle == 270:
    img = cv2.rotate(org_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
elif angle == 180:
    img = cv2.rotate(org_img, cv2.ROTATE_180)
elif angle == 90:
    img = cv2.rotate(org_img, cv2.ROTATE_90_CLOCKWISE)
else:
    img = org_img
cv2.imshow('org', org_img)
cv2.waitKey(0)
cv2.imshow('sf', img)
cv2.waitKey(0)