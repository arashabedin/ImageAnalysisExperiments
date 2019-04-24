from __future__ import print_function
import cv2 as cv2
import numpy as np
import argparse
from numpy.core.umath_tests import inner1d
from matplotlib import pyplot as plt
from scaleInnerContents import scaleInnerContents



def alignImages_sift(im1, im2):
 
    MIN_MATCH_COUNT = 4

    gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)


    ## (2) Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    ## (3) Create flann matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

    ## (4) Detect keypoints and compute keypointer descriptors
    kpts1, descs1 = sift.detectAndCompute(gray1,None)
    kpts2, descs2 = sift.detectAndCompute(gray2,None)



    ## (5) knnMatch to get Top2
    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    canvas = im2.copy()


    if len(good)>MIN_MATCH_COUNT:

        ## (queryIndex for the small object, trainIndex for the scene )
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        ## find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        h,w = im1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))



    ## (9) Crop the matched region from scene
    h,w = im1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
    im2_aligned = cv2.warpPerspective(im2,perspectiveM,(w,h))

    return im2_aligned