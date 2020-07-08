# encoding=utf-8
''' 移除不好的背景：1.宽度小于长度的（width<height); 2.尺寸过小的 '''

import os
import cv2

THRESH_H = 700
THRESH_W = 900

file_to_remove = []
for f in os.listdir('./material/background'):
    fn = os.path.join('./material/background', f)
    img = cv2.imread(fn)
    if img is None:
        print('found error image: ', fn)
        continue
    height, width = img.shape[:2]
    if height < THRESH_H or width < THRESH_W:
        print(fn)
        file_to_remove.append(fn)

for fn in file_to_remove:
    print(fn)
    os.remove(fn)