import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4

training_img = cv2.imread("Images/Im1.jpeg")
training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)

testing_img = cv2.imread("Images/Im4.jpeg")
testing_img = cv2.cvtColor(testing_img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('Training Image', training_img)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(training_img, None)
kp2, des2 = sift.detectAndCompute(testing_img,None)

# Draw keypoints on image training image and display
training_img = cv2.drawKeypoints(training_img, kp1, training_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('Detected Keypoints', training_img)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = training_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    testing_img = cv2.polylines(testing_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(training_img,kp1,testing_img,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'), plt.show()

#cv2.waitKey()
