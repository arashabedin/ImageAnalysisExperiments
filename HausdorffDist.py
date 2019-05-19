import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt
from PIL import Image
from alignImages import alignImages



# def HausdorffDist(A,B):
#             # D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
#             # dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
#             # return(dH)
#             return directed_hausdorff(A, B)[0]

def HausdorffDist(A,B):
            # D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
            # dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
            # return(dH)
            d_forward = directed_hausdorff(A, B)[0]
            d_backward = directed_hausdorff(B, A)[0]
            return max(d_forward, d_backward)

# img = cv2.imread('input_images/box/AT_Translate_2.png')
# img = cv2.imread('input_images/box/box_1_201.jpg')
# img = cv2.imread('input_images/box/AT_Translate_2_re.jpg')
# img = cv2.imread('input_images/box/shapes_1_f.png')
img = cv2.imread('input_images/box/box_1_17_e.jpg')
# img = cv2.imread('input_images/box/Edge_Detected_Box_2_borders.png')
# img = cv2.imread('input_images/box/circle.png')
img = cv2.resize(img,(800,600))
# height, width = img.shape

img2 = cv2.imread('input_images/box/Edge_Detected_Box_2.png')
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
# gray2 = myAligned
gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners2 = cv2.goodFeaturesToTrack(gray2,225,0.01,10)
corners2 = np.int0(corners2)


n1 = np.squeeze(np.asarray(corners))
n2 = np.squeeze(np.asarray(corners2))
distance = HausdorffDist(n1,n2)
print("the final results:")
print(distance)




for i in corners:
    x,y = i.ravel()
    cv2.circle(blank,(x,y),3,255,-1)

for i in corners2:
    x,y = i.ravel()
    cv2.circle(blank2,(x,y),3,255,-1)

textstr = "Hausdorff distance: " + str(distance)

########################################
##### Displaying the final results #####
########################################

plt.subplot(121),plt.imshow(blank,cmap = 'gray')
plt.title('Corners from original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blank2,cmap = 'gray')
plt.title('Corners from wrong drawing'), plt.xticks([]), plt.yticks([]),
plt.imshow(blank2),
plt.text(-500, 800, textstr, fontsize=14)
plt.grid(True)
plt.show()
