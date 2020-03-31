import cv2
import numpy as np
from matplotlib import pyplot as plt
from helpers.corner_detection import maskImage
import imutils

def templateMatching(training_img, testing_img, visualize):
    found = None
    training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)

    training_img = cv2.GaussianBlur(training_img, (7, 7), 0)
    training_edged = cv2.Canny(training_img, 50, 150)
    (tH, tW) = training_edged.shape[:2]

    cv2.rectangle(training_edged, (0, 0), (training_edged.shape[1]-1,training_edged.shape[0]-1), (255, 0, 0), 1)
    cv2.imshow('Training Edged', training_edged)
    cv2.waitKey()

    testing_img = cv2.cvtColor(testing_img, cv2.COLOR_BGR2GRAY)
    testing_blurred = cv2.GaussianBlur(testing_img, (7, 7), 0)
    testing_edged = cv2.Canny(testing_blurred, 50, 150)
    cv2.imshow('Testing Edged', testing_edged)
    cv2.waitKey()

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(testing_edged, width=int(testing_edged.shape[1] * scale))
        r = testing_edged.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        result = cv2.matchTemplate(resized, training_edged, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if visualize:
            # draw a bounding box around the detected region
            clone = np.dstack([resized, resized, resized])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    cv2.rectangle(testing_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", testing_img)
    cv2.waitKey()

def SIFTMatching(training_img, testing_img, method, edge_flag):
    MIN_MATCH_COUNT = 10

    training_img_orig = training_img
    testing_img_orig = testing_img

    training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)
    testing_img = cv2.cvtColor(testing_img, cv2.COLOR_BGR2GRAY)

    testing_img = maskImage(testing_img)
    training_img = maskImage(training_img)

    # gray = np.float32(testing_img)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # dst[dst>0.0001*dst.max()]=[255]
    # cv2.imshow('Harris', dst)
    # cv2.waitKey()


    #cv2.imshow('Training Image', training_img)

    if edge_flag:
        # Adding Canny Filter
        blurred = cv2.GaussianBlur(training_img, (7, 7), 0)
        training_img = cv2.Canny(blurred, 50, 150)

        blurred = cv2.GaussianBlur(testing_img, (7, 7), 0)
        testing_img = cv2.Canny(blurred, 10, 150)

        #cv2.imshow('Modified Training Image', training_img)
        #cv2.waitKey()
        # training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)
        # testing_img = cv2.cvtColor(testing_img, cv2.COLOR_BGR2GRAY)
        #training_img = cv2.cornerHarris(training_img,2,3,0.04)
        #testing_img = cv2.cornerHarris(testing_img,2,3,0.04)

        # training_img = cv2.Laplacian(training_img, cv2.CV_64F)
        # cv2.imshow('Laplacian', training_img)
        # cv2.waitKey()
        # testing_img = cv2.Laplacian(testing_img, cv2.CV_64F)

    if method == "sift":
        detector = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 9999999)
    elif method == "surf":
        detector = cv2.xfeatures2d.SURF_create(4000)
        #detector.setUpright(True)
    elif method == "orb":
        detector = cv2.ORB_create()

    # find the keypoints and descriptors with detector
    kp1, des1 = detector.detectAndCompute(training_img, None)
    kp2, des2 = detector.detectAndCompute(testing_img, None)

    # training_img_orig = training_img
    # testing_img_orig = testing_img
    # Draw keypoints on image training image and display
    training_img = cv2.drawKeypoints(training_img_orig, kp1, training_img_orig, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    testing_img = cv2.drawKeypoints(testing_img_orig, kp2, testing_img_orig, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Detected Keypoints', testing_img_orig)
    cv2.waitKey()
    cv2.imshow('Detected Keypoints', training_img_orig)
    cv2.waitKey()

    if method == "sift" or "surf":
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    elif method == "orb":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

    #This is currently overriding FLANN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

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


############################################################################################################################


train_img = cv2.imread("../Images/Sign2.jpeg")

#cv2.rectangle(train_img, (0, 0), (train_img.shape[1]-1,train_img.shape[0]-1), (255, 255, 255), 1)
#cv2.imshow('train img', train_img)
#cv2.waitKey()

test_img = cv2.imread("../Images/IMG_200.jpeg")
#test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#templateMatching(train_img, test_img, False)

SIFTMatching(train_img, test_img, "sift", False) # This + canny + contrast threshold of 0.01 works well!

#SIFTMatching(train_img, test_img, "surf", False) # This + canny + contrast threshold of 0.01 works well!



camera = cv2.VideoCapture("../Images/Vid.mov")

while True:
    train_img = cv2.imread("../Images/Sign2.jpeg")

    _, frame = camera.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame,1)
    SIFTMatching(train_img, frame, "sift", False)

cv2.waitKey()
