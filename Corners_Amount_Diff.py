import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt
from PIL import Image
from alignImages import alignImages


img = cv2.imread('input_images/box_1_201.jpg')
img = cv2.resize(img,(800,600))
# height, width = img.shape

img2 = cv2.imread('input_images/shapes_1_e2.jpeg')
img2 = cv2.resize(img2,(800,600))


myAligned = alignImages(img, img2)

height, width, channels = img.shape
print(height)
print(width)

#This time we would add the corners to a white blank image
blank = np.zeros([height,width,3],dtype=np.uint8)
blank.fill(255)
blank2 = np.zeros([height,width,3],dtype=np.uint8)
blank2.fill(255)
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,225,0.01,10)
print(corners)

corners = np.int0(corners)
# print(corners)

# gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners2 = cv2.goodFeaturesToTrack(myAligned,225,0.01,10)
corners2 = np.int0(corners2)


for i in corners:
    x,y = i.ravel()
    cv2.circle(blank,(x,y),3,255,-1)

for i in corners2:
    x,y = i.ravel()
    cv2.circle(blank2,(x,y),3,255,-1)

textstr = "Corners Amounts Difference: " + str(abs(len(corners)-len(corners2)))

########################################
##### Displaying the final results #####
########################################

plt.subplot(121),plt.imshow(blank,cmap = 'gray')
plt.title(' Corners from original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blank2,cmap = 'gray')
plt.title('Corners from aligned drawing'), plt.xticks([]), plt.yticks([]),
plt.imshow(blank2),
plt.text(-800, 680, "Total corners extracted: "+ str(len(corners)), fontsize=12)
plt.text( 150, 680, "Total corners extracted: "+ str(len(corners2)), fontsize=12)

plt.text(-400, 800, textstr, fontsize=14)
plt.grid(True)
plt.show()
