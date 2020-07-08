import os
import cv2
import random


# img = cv2.imread('./material/test.jpg')

# img_erode = cv2.erode(img, (5, 5), iterations=1)

# img_dilate = cv2.dilate(img_erode, (5, 5))

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, (5, 5))
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5, 5))

# cv2.imshow('src', img)
# cv2.imshow('erode', img_erode)
# cv2.imshow('dilate', img_dilate)
# cv2.imshow('opening', opening)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)

methods = ('none', 'erode', 'dilate', 'open', 'close', 'erode2dilate', 'dilate2erode')
def randon_morphology(img):
    method = random.choice(methods)
    kernel = (random.choice((3,5)), ) * 2
    if kernel[0] == 3:
        iterations = random.randint(0,2)
    else:
        iterations = 0
    if method == 'erode':
        dst = cv2.erode(img, kernel, iterations=iterations)
    elif method == 'dilate':
        dst = cv2.dilate(img, kernel, iterations=iterations)
    elif method == 'open':
        dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif method == 'close':
        dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif method == 'erode2dilate':
        dst = cv2.erode(img, kernel, iterations=iterations)
        dst = cv2.dilate(dst, kernel, iterations=iterations)
    elif method == 'dilate2erode':
        dst = cv2.dilate(img, kernel, iterations=iterations)
        dst = cv2.erode(dst, kernel, iterations=iterations)
    else:
        dst = img
    return dst

