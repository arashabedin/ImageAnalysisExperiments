import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('input_photos/CroquisModel.png',0)
edges = cv2.Canny(img,20,120)
edges = cv2.bitwise_not(edges)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

