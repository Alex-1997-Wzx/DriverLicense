''' data_generator.py生成的标注数据是4个点的四边形数据，而PSENET需要的是14点标注的数据，
    关于14点标注法，参考论文：《Detecting Curve Text in the Wild New Dataset and New Solution》
    本脚本用于将data_generator.py生成的标注转化为14点标注格式。
'''
import os
import cv2
import numpy as np
from tqdm import tqdm

root_label_dir = './gen/labels'
ctw_label_dir = './gen/labels_ctw'
# root_image_dir = './gen/original'
# root_label_dir = './data/labeled/txt'
# ctw_label_dir = './data/labeled/txt_ctw'

if not os.path.exists(ctw_label_dir):
    os.mkdir(ctw_label_dir)

for f in tqdm(os.listdir(root_label_dir)):
    ctw_text = []

    fn = os.path.join(root_label_dir, f)
    # print(fn)
    # img_fn = cv2.imdecode(np.fromfile(os.path.join(root_image_dir, f[:-4]+'.jpg'), dtype=np.uint8), cv2.IMREAD_COLOR)
    with open(fn, 'r', encoding='utf-8') as tf:
        lines = tf.readlines()
        # print(type(lines))
        for line in lines:
            lb = line.split(',')
            x0, y0, x1, y1, x2, y2, x3, y3 = [int(k) for k in lb[:8]]
            # print(lb[8])
            if y0 > y1:
                newline = [x0, y1, x2, y3]
                # 上方7个点
                base_x0, base_y0, base_w, base_h = x0, y0, x1-x0, y0-y1
                step_x = base_w // 6
                step_y = base_h // 6
                for i in range(7):
                    tmp_x = base_x0 + i * step_x - x0  # x0: box 基准点
                    tmp_y = base_y0 - i * step_y - y1  # y1: box 基准点
                    newline.extend([tmp_x, tmp_y])
                    # cv2.line(img_fn, (tmp_x, tmp_y), (tmp_x + 3, tmp_y+3), (0, 0, 255), 3)
                # 下方7个点
                base_x0, base_y0, base_w, base_h = x2, y2, x2-x3, y3-y2
                step_x = base_w // 6
                step_y = base_h // 6
                for i in range(7):
                    tmp_x = base_x0 - i * step_x - x0  # x0: box 基准点
                    tmp_y0 = base_y0 + i * step_y - y1  # y1: box 基准点
                    newline.extend([tmp_x, tmp_y0])
                    # cv2.line(img_fn, (tmp_x, tmp_y0), (tmp_x+3, tmp_y0+3), (0, 0, 255), 3)
            else:
                newline = [x3, y0, x1, y2]
                # 上方7个点
                base_x0, base_y0, base_w, base_h = x0, y0, x1-x0, y1-y0
                step_x = base_w // 6
                step_y = base_h // 6
                for i in range(7):
                    tmp_x = base_x0 + i * step_x - x3  # x0: box 基准点
                    tmp_y = base_y0 + i * step_y - y0  # y1: box 基准点
                    newline.extend([tmp_x, tmp_y])

                # 下方7个点
                base_x0, base_y0, base_w, base_h = x2, y2, x2-x3, y2-y3
                step_x = base_w // 6
                step_y = base_h // 6
                for i in range(7):
                    tmp_x = base_x0 - i * step_x - x3  # x0: box 基准点
                    tmp_y = base_y0 - i * step_y - y0  # y1: box 基准点
                    newline.extend([tmp_x, tmp_y])

            newline = ','.join([str(l) for l in newline])
            ctw_text.append(newline)
            # break
    # print(ctw_text)
    target_fn = os.path.join(ctw_label_dir, f)
    with open(target_fn, 'w', encoding='utf-8') as tf:
        tf.write('\n'.join(ctw_text))
    # img_fn = cv2.resize(img_fn, (1000, 700))
    # cv2.imshow('img', img_fn)
    # cv2.waitKey(0)
    # break