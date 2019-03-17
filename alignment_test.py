from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from numpy.core.umath_tests import inner1d
from matplotlib import pyplot as plt
from alignImages import alignImages
from scaleInnerContents import scaleInnerContents

img = cv.imread('input_images/box/AT_Translate_2.png',0)
img= cv.resize(img,(800,600))
height, width = img.shape

img2 = cv.imread('input_images/box/Edge_Detected_Box_2.png',0)
img2 = cv.resize(img2,(800,600))


# myAligned = alignImages(img, img2)
# rows,cols = img.shape
# M = np.float32([[1,0,200],[0,1,150]])
# myAligned = cv.warpAffine(img,M,(cols,rows))



cropped2 = scaleInnerContents(img,1.2)
height, width = cropped2.shape
print(height)
print(width) 

cv.imshow('aft', cropped2)
cv.waitKey()




corners = cv.goodFeaturesToTrack(myAligned,225,0.01,10)
corners = np.int0(corners)

blank = np.zeros([height,width,3],dtype=np.uint8)
blank.fill(255)

for i in corners:
    x,y = i.ravel()
    cv.circle(blank,(x,y),3,255,-1)


########################################
##### Displaying the final results #####
########################################
# def HausdorffDist(A,B):
#         D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
#         dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
#         return(dH)
# corners = cv.goodFeaturesToTrack(img2,225,0.01,10)
# corners2 = cv.goodFeaturesToTrack(myAligned,225,0.01,10)
# n1 = np.squeeze(np.asarray(corners))
# n2 = np.squeeze(np.asarray(corners2))

# distance = HausdorffDist(n1,n2)
# print("distance")
# print(distance)

myAligned = cv.cvtColor(myAligned,cv.COLOR_GRAY2RGB)


# plt.subplot(121),plt.imshow(img2,cmap = 'gray')
# plt.title('Main Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(myAligned,cmap = 'gray')
# plt.title('Aligned Image'), plt.xticks([]), plt.yticks([])
# plt.imshow(myAligned),plt.show()
