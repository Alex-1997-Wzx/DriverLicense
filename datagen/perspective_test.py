''' 透视变换测试脚本，不用在生产环境 '''
import os
import cv2
import numpy as np
import random


def get_perspective_matrix(img, points):
    h, w = img.shape[:2]
    h_bias_left = random.randint(15, 20) * h // 100
    h_bias_right = random.randint(15, 20) * h // 100
    w_bias_left = random.randint(15, 20) * w // 100
    w_bias_right = random.randint(15, 20) * w // 100

    # 左图中画面中的点的坐标 四个
    pts1 = np.float32(points)
    # 变换到新图片中，四个点对应的新的坐标 一一对应
    # 随机选择两个点进行透视变换
    # seed = random.randint(0, 3)
    seed = 0
    new_ps = []
    # for i, xy in enumerate(points):
        # if i == seed or (seed+1)%4 == i:
    if seed == 0:
        new_ps.append((points[0][0]+random.randint(10, 30), points[0][1] + h_bias_left))
        new_ps.append((points[1][0]-random.randint(10, 30), points[1][1] + h_bias_right))
        new_ps.append(points[2])
        new_ps.append(points[3])
    elif seed == 1:
        new_ps.append(points[0])
        new_ps.append((points[1][0] - w_bias_left, points[1][1] + random.randint(10, 30)))
        new_ps.append((points[2][0] - w_bias_right, points[2][1] - random.randint(10, 30)))
        new_ps.append(points[3])
    elif seed == 2:
        new_ps.append(points[0])
        new_ps.append(points[1])
        new_ps.append((points[2][0]-random.randint(10, 30), points[2][1] - h_bias_left))
        new_ps.append((points[3][0]+random.randint(10, 30), points[3][1] - h_bias_right))
    else:
        new_ps.append((points[0][0]+w_bias_left, points[0][1] + random.randint(10, 30)))
        new_ps.append(points[1])
        new_ps.append(points[2])
        new_ps.append((points[3][0]+w_bias_right, points[3][1] - random.randint(10, 30)))

    pts2 = np.float32(new_ps)

    # 生成变换矩阵
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # 进行透视变换
    # dst = cv2.warpPerspective(img,M,(300,300))
    return M



img = cv2.imread(r'C:\temp\pad.jpg')
h,w = img.shape[:2]
ps = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
pM = get_perspective_matrix(img, ps)
# inner = cv2.perspectiveTransform(np.array([[inner]]), pM, (target_w, target_h))
img = cv2.warpPerspective(img, pM, (w, h))

# cv2.imshow('adsf', img)
# cv2.waitKey(0)
# cv2.imwrite(r'C:\temp\pad2.jpg', img)

# old_points = [328, 124]
a = np.array([[328, 124], [348, 120], [118,178], [690,86]], dtype='float32')
print('shape: ', a.shape)
# h = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
a = np.array([a])
print('shape: ', a.shape)

pointsOut = cv2.perspectiveTransform(a, pM)

print(type(pointsOut))
print(pointsOut)
print(pointsOut.shape)
print(pointsOut[0][1][0])
print(pointsOut[0][1][1])