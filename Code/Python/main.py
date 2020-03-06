#import helpers
import imutils
import cv2
import numpy as np

def cornerIn(curr, image_points):
	for x, y in image_points:
		if abs(x - curr[0]) < 25 and abs(y - curr[1]) < 25:
			return True
	return False

def getCorners(frame,last_aspectRatio):
	corners = np.empty([0, 2], dtype=np.float32)
	current_corners = np.empty([0, 2], dtype=np.float32)

	detected = False
	status = "No Targets"

	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 50, 150)
	#cv2.imshow("edged",edged)

	# find contours in the edge map
	cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

	# ensure that the approximated contour is "roughly" rectangular
		if len(approx) == 4:
			# compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)

			# compute the solidity of the original contour
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			# compute whether or not the width and height, solidity, and
			# aspect ratio of the contour falls within appropriate bounds
			keepDims = w > 25 and h > 25
			keepSolidity = solidity > 0.9
			keepAspectRatio = aspectRatio >= 1.4 and aspectRatio <= 1.8

			detected = keepDims and keepSolidity and keepAspectRatio

			# ensure that the contour passes all our tests
			if detected and not cornerIn([x, y], current_corners):
				current_corners = np.append(current_corners, np.array([[x, y]]), axis=0)
				approx = np.squeeze(approx, axis = 1)
				#print("approx: ", np.shape(approx), " corners: ", np.shape(corners))
				corners = np.vstack([corners, approx])

				# draw an outline around the target and update the status text
				cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
				last_aspectRatio = aspectRatio
				status = "Target(s) Acquired"
				# compute the center of the contour region and draw the crosshairs
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] / (M["m00"] + 1e-7)), int(M["m01"] / (M["m00"] + 1e-7)))
				cv2.circle(frame, (cX, cY), 5, (0, 0, 255), thickness=-1, lineType=8, shift=0)

	# draw the status text on the frame
	cv2.putText(frame, status + " - Aspect Ratio: " + str('%0.3f' % last_aspectRatio), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the frame and record if a key is pressed
	#cv2.imshow("Corner Detection", frame)
	return frame, detected, corners, last_aspectRatio


camera = cv2.VideoCapture(0)
last_aspectRatio = 0

def findDoubleBox(arr):
	n_rects = len(arr)/2

	if(n_rects == 1):
		return arr

def estimateCameraPose(objp, corners, mtx, dist):

	ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

	tvec = np.squeeze(tvec)

	if ret:
		Rt,jacob = cv2.Rodrigues(rvec)
		R = Rt.transpose()
		#print("R",Rt)
		pos = tvec.dot(-Rt)
		print("Distance: ",np.linalg.norm(pos))
		print("pos",pos)

	return pos

#worldPoints = createRectWorldPoints(25, 270, 190, 220, 140)
def createRectWorldPoints(off, w1, h1, w2, h2):
	worldPnts = np.zeros((8,3), dtype=np.float32)

	worldPnts[0] = np.array([0,0,0])
	worldPnts[1] = np.array([0,h1,0])
	worldPnts[2] = np.array([w1,0,0])
	worldPnts[3] = np.array([w1,h1,0])

	worldPnts[4] = np.array([off,off,0])
	worldPnts[5] = np.array([off,off+h2,0])
	worldPnts[6] = np.array([off+w2,off,0])
	worldPnts[7] = np.array([off+w2,off+h2,0])

	return worldPnts

#This function should be deleted
'''
def sortCorners(corners):
	out_corners = np.array((4,3))
	
	out_corners = corners[corners[:,1].argsort()]
	out_corners[(0,1),:] = corners[corners[(0,1),0].argsort()]
	out_corners[(2,3),:] = corners[2+corners[(2,3),0].argsort()]
	out_corners[(3,4),:] = corners[3+corners[(3,4),0].argsort()]
	out_corners[(5,6),:] = corners[5+corners[(5,6),0].argsort()]
	#out_corners[0:1,:] = corners[corners[0:1,0].argsort()]
	#print(corners[corners[(0,1),0].argsort()])
	#out_corners[2:3,:] = corners[corners[2:3,0].argsort()]
	return out_corners
'''
def trackRect(frame,x,y,w,h):	#We can probably change this function to pass roi directly
	track_window = (x, y, w, h)
	roi = frame[y:y + h, x:x + w]
	hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
	roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)	#This could probably be calculated somewhere else
	dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

	# apply CamShift to get the new location
	ret, track_window = cv2.CamShift(dst, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1))

	# Draw it on image
	x, y, w, h = track_window
	return x,y,w,h

'''

def sortCorners(corners):
	corners = corners[corners[:,0].argsort()]

	corners[0:2] = corners[corners[0:2,1].argsort()]
	corners[2:4] = corners[2+corners[2:4,1].argsort()]
	corners[4:6] = corners[4+corners[4:6,1].argsort()]
	corners[6:8] = corners[6+corners[6:8,1].argsort()]
	 
	corners = np.insert(corners, 2, np.array([corners[-2], corners[-1]]), axis=0)
	corners = np.delete(corners, [8, 9], axis=0) 
	
	return corners
# ~~~~~~~~~~~~~~Start of MAAAAIIIINNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mtx = np.load("mtx.npy")
dist = np.load("dist.npy")
last_aspectRatio = 0

#camera = cv2.VideoCapture(0)

#while True:
	# grab the current frame and initialize the status text

	#_, frame = camera.read()
frame = cv2.imread("Image-900mm.jpeg")
frame_to_show, corners, last_aspectRatio = getCorners(frame, last_aspectRatio)

#if detected:
corners = sortCorners(corners)
print("Corners:\n",corners)

#Count the corner inside the image
counter = 0
for x, y in corners:
	counter+=1
	cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=8, shift=0)
	cv2.putText(frame, str(counter), (int(x + 12), int(y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

worldPoints = createRectWorldPoints(22, 250, 170, 207, 123)
print("World Points:\n", worldPoints)

pos = estimateCameraPose(worldPoints, corners, mtx, dist)
print("Distance: ", np.linalg.norm(pos))
cv2.imshow('Frame To Show',frame_to_show)

key = cv2.waitKey(0)

# cleanup the camera and close any open windows
#camera.release()
cv2.destroyAllWindows()
