import numpy as np
import cv2 as cv
import glob
import math
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)*25
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('Images/Image-0*.jpeg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img',gray)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx:\n", mtx)
print("dist:\n", dist)


#Camera Parameters (Extracted from matlab calibration)
mtx = np.array([[1230.23369751200,0,603.198823692201], [0,1229.55591395379,587.295825753351],[0,0,1]])
dist = np.array([0.225126643359879,-0.881346132038369,0,0])

boardShape = (8,5)

#Read the Image
img = cv.imread('Images/Image-60cm.jpeg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
img = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imshow('img', img)
cv.waitKey()

print(np.shape(img))

# Find the chess board corners
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.waitKey()
ret, corners = cv.findChessboardCorners(gray, boardShape , None)

if ret:
	print("Corners Found")	
	corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	imgpoints.append(corners)
	# Draw and display the corners
	cv.drawChessboardCorners(img, boardShape , corners, ret)
	cv.imshow('img', img)
	cv.waitKey()
else:
	print("Corners Not Found!")


#h, w = img.shape[:2]
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
#dst = cv.undistort(img, mtx, dist, None, mtx)

# crop the image
#x, y, w, h = roi
#dst = dst[y:y + h, x:x + w]
#cv.imwrite('calibresult.png', dst)
#cv.imshow('dst',dst)
#cv.waitKey()

objpoints.append(objp)
print("Object Points:\n",np.shape(objp),"\n",objp)
print("Image Points:\n",np.shape(corners),"\n",corners)

#getting distance without using undistorted results
# TODO: change the indexes according to the image we are undistorting
ret, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)

if ret:
	d = np.linalg.norm(tvec)
	print(d)