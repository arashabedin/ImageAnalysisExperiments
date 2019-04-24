import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

original_img = cv2.imread('input_images/box/bluebox.JPG')
RGB_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
grayscale_img = cv2.cvtColor(original_img ,cv2.COLOR_BGR2GRAY)

#enhancing the contrast
def contrastEnhancer(img): 
    hist,bins = np.histogram(img.flatten(),256,[0,80])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]


# edges = cv2.Canny(contrastEnhancer(grayscale_img) ,30,60)
edges = cv2.Canny(grayscale_img ,30,60)
edges = cv2.bitwise_not(edges)


plt.subplot(121),plt.imshow(RGB_img,cmap = 'gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.title('Canny edge detected'), plt.xticks([]), plt.yticks([])
plt.imshow(edges,cmap = 'gray'),plt.show()
