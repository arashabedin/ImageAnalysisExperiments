import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#######################################
##### Shi Tomasi Corner Detection #####
#######################################

img = cv2.imread('results/Canny-Edge-Detected.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,225,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


########################################
##### Displaying the final results #####
########################################

plt.title('Shi-Tomasi Detector'), plt.xticks([]), plt.yticks([])
plt.imshow(img),plt.show()
