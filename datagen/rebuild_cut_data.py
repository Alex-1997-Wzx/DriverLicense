
''' 改变cut词条的长和宽, 后面发现没必要，在CRNN里面参数可以调节 '''
import os
import cv2
import numpy as np


TARGET_W = 380
TARGET_H = 22
root_dir = './gen/cut'
target_dir = './gen/cut_rebuild'

if not os.path.exists(root_dir):
    print(root_dir + '  not exists!')
    os._exit()

if not os.path.exists(target_dir):
    print('create target dir: ' + target_dir)
    os.mkdir(target_dir)


blank_image = np.zeros((TARGET_H,TARGET_W,3), np.uint8)
blank_image = (255 - blank_image)

files = os.listdir(root_dir)
len_files = len(files)
file_to_save = []
for i, f in enumerate(files):
    if (i+1) % 100 == 0:
        print('{} / {}'.format(i, len_files))
        for im, fn in file_to_save:
            cv2.imencode('.jpg', im)[1].tofile(fn)
    fn = os.path.join(root_dir, f)
    img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # if h/TARGET_H > w/TARGET_W:
    #     new_w = TARGET_W
    #     new_h = int(h * new_w / w)
    # else:
    new_h = TARGET_H
    new_w = int(w * new_h / h)
    new_img = cv2.resize(img, (new_w, new_h))

    module = blank_image.copy()
    try:
        module[:, :new_w, :] = new_img
    except Exception as e:
        print(str(e))
        print(f)
        continue

    file_to_save.append((module, os.path.join(target_dir, f)))


for im, fn in file_to_save:
    cv2.imencode('.jpg', im)[1].tofile(fn)
print('\nconvert finished.')