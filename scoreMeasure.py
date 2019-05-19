import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt
from alignImages import alignImages
from numpy import linalg as LA
from chamferDist import  chamfer_distance_numpy


def HausdorffDist(A,B):
    d_forward = directed_hausdorff(A, B)[0]
    d_backward = directed_hausdorff(B, A)[0]
    return max(d_forward, d_backward)

def matchContours(im1, im2):
    im1 =  cv2.bitwise_not(im1)
    im2 =  cv2.bitwise_not(im2)
    ret = cv2.matchShapes(im1,im2,1,0.0)
    return round(ret,6)
    
# img = cv2.imread('input_images/box/shapes_borders.png')
img = cv2.imread('input_images/box/AT_Translate_2_re.jpg')
# img = cv2.imread('input_images/box/shapes_1_f.png')
# img = cv2.imread('input_images/box/box_1_17_e.jpg')
# img = cv2.imread('input_images/box/circle.png')
# img = cv2.imread('input_images/box/shapes_1_e2.jpeg')
img = cv2.resize(img,(800,600))

# img2 = cv2.imread('input_images/box/shapes_1_f.png')
img2 = cv2.imread('input_images/box/shape_main.png')

img2 = cv2.resize(img2,(800,600))
# img = img2
# myAligned = alignImages(img, img2_1)


height, width, channels = img.shape

gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,120,0.01,10)
corners = np.int0(corners)

# gray2 = myAligned
gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners2 = cv2.goodFeaturesToTrack(gray2,120,0.01,10)
corners2 = np.int0(corners2)


n1 = np.squeeze(np.asarray(corners))
n2 = np.squeeze(np.asarray(corners2))




# champer = chamfer_distance_numpy(corners,corners2)
len1 = len(n1)
len2 = len(n2)
distance = HausdorffDist(n1,n2)
contourDiff =  matchContours(gray, gray2)
# print(((champer - (champer * 4/5))*2))
print(contourDiff*5000)
print(distance * 4)
if distance < 80:
    distance = distance - ((80 - distance) * 6)
    if distance < 0:
        distance = 0

results = (distance* 4) + (contourDiff*5000) + (abs(len1-len2)* 0.2) 
print(results)
print("___")

worstResult = 1000
score = int(((worstResult-results)/worstResult)*100)
if score < 0 : score = 0 
# print(champer)
print(contourDiff)
print(distance)
print(abs(len1-len2))
textstr = "score: " + str(score) 


########################################
##### Displaying the final results #####
########################################

plt.subplot(121),plt.imshow(img2,cmap = 'gray')
plt.title('Edge detected original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = 'gray')
plt.title('Wrong drawing'), plt.xticks([]), plt.yticks([]),
plt.imshow(img),
plt.text(-190, 800, textstr, fontsize=16)
plt.grid(True)
plt.show()
