import cv2
import numpy as np

def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    return img_crop


if __name__ == '__main__':
    # generate image
    img = np.zeros((1000, 1000), dtype=np.uint8)
    img = cv2.line(img, (400, 400), (511,511), 1, 120)
    img = cv2.line(img, (300, 300), (700,500), 1, 120)

    # find contours / rectangle
    contours,_ = cv2.findContours(img, 1, 1)
    rect = cv2.minAreaRect(contours[0])

    # crop
    img_croped = crop_minAreaRect(img, rect)

    # show
    import matplotlib.pylab as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_croped)
    plt.show()