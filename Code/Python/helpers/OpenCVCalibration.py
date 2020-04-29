import numpy as np
import cv2 as cv
import glob
import math

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

calibrate = False;

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)*20
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('../Images/server/*.jpeg') #Try to use only the first 12 images



parametersLoaded = False
try:
	mtx = np.load("../data/mtx.npy")
	dist = np.load("../data/dist.npy")
	parametersLoaded = True
except:
	print("Parameters not loaded")

if calibrate or not parametersLoaded:
	print("Calibrating")
	gray = cv.imread("../Images/server/*.jpeg")
	for fname in images:
	    img = cv.imread(fname)
	    #img = cv.resize(img,(1200,1200))
	    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	    #cv.imshow('img',gray)
	    # Find the chess board corners
	    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
	    # If found, add object points, image points (after refining them)
	    print(ret)
	    if ret == True:
	        objpoints.append(objp)
	        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	        corners = corners2.astype('float32')
	        imgpoints.append(corners2)
	        # Draw and display the corners
	        cv.drawChessboardCorners(img, (8,5), corners2, ret)
	        cv.imshow('img', img)
	        cv.waitKey(500)
	cv.destroyAllWindows()

	#print("GRAY: ",gray.shape[::-1])
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],mtx,dist,flags = cv.CALIB_USE_INTRINSIC_GUESS)
	np.save("../data/mtx",mtx)
	np.save("../data/dist",dist)


boardShape = (8,5)

#url = "http://admin:admin@192.168.43.1:8080/video"
#camera = cv.VideoCapture(url)

#_, img = camera.read();

#Read the Image
img = cv.imread('../Images/server/img3`.jpeg') 
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#cv.imshow('Original', img)
#cv.waitKey()
mtx = newcameramtx
# img = cv.undistort(img, mtx, dist, None, mtx)
# cv.imshow("Undistorted", img)
# cv.waitKey()

print("Image shape: ",np.shape(img))

# Find the chess board corners
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.waitKey()
ret, corners = cv.findChessboardCorners(gray, boardShape , None)

if ret:
	print("Corners Found")	
	#corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	#imgpoints.append(corners)

	# Draw and display the corners
	cv.drawChessboardCorners(img, boardShape , corners, ret)
	cv.imshow('img', img)
	cv.waitKey()
else:
	print("Corners Not Found!")

print("Object Points:\n",np.shape(objp),"\n",objp)
#print("Image Points:\n",np.shape(corners),"\n",corners)
#getting distance without using undistorted results
ret, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)
print("SolvePNP Worked: ", ret)

axis = np.float32([[12.5,0,0], [0,12.5,0], [0,0,-12.5]]).reshape(-1,3)
imgpts, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)

img = draw(img,corners,imgpts)
cv.imshow('img',img)
cv.waitKey()

tvec = np.squeeze(tvec)
print("tvec",tvec)

#this might be wrong
if ret:
	Rt,jacob = cv.Rodrigues(rvec)
	R = Rt.transpose()
	print("R",Rt)
	pos = tvec.dot(-Rt)
	print("Distance: ",np.linalg.norm(pos))
	print("pos",pos)