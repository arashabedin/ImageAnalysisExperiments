import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

###################################
##### Harris Corner Detection #####
###################################

img = cv2.imread('results/Canny-Edge-Detected.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]


########################################
##### Displaying the final results #####
########################################

plt.title('Harris Corner Detector'), plt.xticks([]), plt.yticks([])
plt.imshow(img),plt.show()
