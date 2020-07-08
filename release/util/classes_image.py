import os
import cv2
import numpy as np
import tqdm
import shutil


root_dir = r'D:\xiashu\OCR\jutze\data\origin'
output_dir = './data/test_origin'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 同时满足以下条件视为同一张图片
# width相差不超多5个像素点
# height相差不超过5个像素点

images = []

for sub in ['juzi1199-cuojian', 'juzi1776']:
    d = os.path.join(root_dir, sub)
    for f in tqdm.tqdm(os.listdir(d)):
        fn = os.path.join(d, f)
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
        not_found = True
        for w, h, _ in images:
            if abs(w - img.shape[0]) <= 5 and abs(h - img.shape[1]) <= 5:
                not_found = False
                break
        if not_found:
            images.append((img.shape[0], img.shape[1], fn))

print('found images: ', len(images))

for w, h, fn in tqdm.tqdm(images):
    f = os.path.split(fn)[-1]
    new_f = os.path.join(output_dir, f)
    shutil.copyfile(fn, new_f)