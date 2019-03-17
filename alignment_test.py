from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from numpy.core.umath_tests import inner1d
from matplotlib import pyplot as plt
from alignImages import alignImages


img = cv.imread('input_images/box/AT_Translate_2.png',0)
img= cv.resize(img,(800,600))
height, width = img.shape

img2 = cv.imread('input_images/box/Edge_Detected_Box_2.png',0)
img2 = cv.resize(img2,(800,600))


myAligned = alignImages(img, img2)
# corners = cv.goodFeaturesToTrack(myAligned,225,0.01,10)
# corners = np.int0(corners)

# blank = np.zeros([height,width,3],dtype=np.uint8)
# blank.fill(255)

# for i in corners:
#     x,y = i.ravel()
#     cv.circle(blank,(x,y),3,255,-1)
########################################
##### Displaying the final results #####
########################################


plt.subplot(121),plt.imshow(img2,cmap = 'gray')
plt.title('Main Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(myAligned,cmap = 'gray')
plt.title('Aligned Image'), plt.xticks([]), plt.yticks([])
plt.imshow(myAligned),plt.show()
