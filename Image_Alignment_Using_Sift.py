from cv2 import cv2, countNonZero, cvtColor
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy.core.umath_tests import inner1d
from alignImages_sift import alignImages_sift




######################################
##### Image Alignment Using Sift #####
######################################

img1 = cv2.imread('results/Edge_Detected_Box.png')
img2 = cv2.imread('input_images/box/Edge_Detected_Box_smaller.png')

##### Image Alignment Using Sift #####
##### Image Alignment Using Sift #####
img1 = cv2.resize(img1,(800,600))
img2 = cv2.resize(img2,(800,600))
img2_2 = img2



# im2_aligned = cv2.warpAffine(img,M,(w,h))
im2_aligned = alignImages_sift(img1, img2)



########################################
##### Displaying the final results #####
########################################

plt.subplot(121),plt.imshow(img1,cmap = 'gray')
plt.title('Orignial image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),
plt.title('Aligned drawing'), plt.xticks([]), plt.yticks([])
plt.imshow(im2_aligned),plt.show()
