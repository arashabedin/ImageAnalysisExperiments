import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('input_images/box/model_edited.jpg',0)

#enhancing the contrast
def contrastEnhancer(img): 
    hist,bins = np.histogram(img.flatten(),256,[0,80])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]


edges = cv2.Canny(contrastEnhancer(img) ,60,120)
edges = cv2.bitwise_not(edges)

plt.imshow(edges,cmap = 'gray')

plt.show()

