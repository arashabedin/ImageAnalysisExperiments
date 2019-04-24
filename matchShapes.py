import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('input_images/box/shapes_1_e.png',0)
img2 = cv2.imread('input_images/box/box_1_201.jpg',0)
img1 =  cv2.bitwise_not(img1)
img2 =  cv2.bitwise_not(img2)
print(img1.dtype)

def getBordered(image, width):
    bg = np.zeros(image.shape)
    _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > biggest:
            biggest = area
            bigcontour = contour
    return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(np.uint8)

img1 = getBordered(img1, 5)
print(img1.dtype)
# img1 = np.array(img1, dtype=np.uint8)
img2 = getBordered(img2, 5)

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# img2 = np.array(img2, dtype=np.uint8)

# img1 = cv2.Canny(img1 ,60,120)
# img2 = cv2.Canny(img2 ,60,120)

# edges2 = cv2.Canny(img2 ,60,120)
# edges2 = cv2.bitwise_not(edges2)

# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# plt.title('inverted'), plt.xticks([]), plt.yticks([])
# plt.imshow(img1),plt.show()

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
_,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
_,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]
# print(cnt2)

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print ret