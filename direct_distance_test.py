import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt
from alignImages import alignImages
from numpy import linalg as LA
from direct_distance import directDistance


    
# img = cv2.imread('input_images/box/box_1_17_e.jpg')
img = cv2.imread('input_images/box/AT_Translate_2_re.jpg')
# img = cv2.imread('input_images/box/shapes_1_f.png')
# img = cv2.imread('input_images/box/box_1_17_e.jpg')
# img = cv2.imread('input_images/box/circle.png')
# img = cv2.imread('input_images/box/box_1_200.jpg')
# img = cv2.imread('input_images/box/shapes_borders.png')

# img = cv2.resize(img,(800,600))

img2 = cv2.imread('input_images/box/Edge_Detected_Box_3.png')
# img2 = cv2.imread('input_images/box/shape_main.png')

img2 = cv2.resize(img2,(800,600))
# img = img2
myAligned = alignImages(img, img2)


height, width, channels = img.shape

gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,120,0.01,10)
# corners = np.int0(corners)

gray2 = myAligned
# gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# corners2 = cv2.goodFeaturesToTrack(gray2,120,0.01,10)
# corners2 = np.int0(corners2)

blank = np.zeros([height,width,3],dtype=np.uint8)
blank.fill(255)
blank2 = np.zeros([height,width,3],dtype=np.uint8)
blank2.fill(255)
direct , corners , corners2 = directDistance(gray,gray2)

for i in corners:
    x,y = i.ravel()
    cv2.circle(blank,(x,y),3,255,-1)

for i in corners2:
    x,y = i.ravel()
    cv2.circle(blank2,(x,y),3,255,-1)

# n1 = np.squeeze(np.asarray(corners))
# n2 = np.squeeze(np.asarray(corners2))





textstr = "distance: " + str(direct) 


########################################
##### Displaying the final results #####
########################################

plt.subplot(121),plt.imshow(blank,cmap = 'gray')
plt.title('Corners from original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blank,cmap = 'gray')
plt.title('Corners from aligned drawing'), plt.xticks([]), plt.yticks([]),
plt.imshow(blank2),
plt.text(-390, 800, textstr, fontsize=16)
plt.grid(True)
plt.show()
