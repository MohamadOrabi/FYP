#import helpers
import imutils
import cv2
import numpy as np

def getCorners(frame,last_aspectRatio):
	corners = np.empty([0, 2], dtype=int)
	detected = False
	status = "No Targets"

	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 50, 150)
	#cv2.imshow("edged",edged)

	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
			keepAspectRatio = aspectRatio >= 1.3 and aspectRatio <= 1.5

			detected = keepDims and keepSolidity and keepAspectRatio

			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				#print(x,y,w,h)
				#current_corners = np.array([x,y,w,h])
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

def estimateCameraPose(corners, objp, mtx, dist):

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

def createRectWorldPoints(w,h):
	worldPnts = np.zeros((4,3))
	worldPnts[1] = np.array([w,0,0])
	worldPnts[2] = np.array([0,h,0])
	worldPnts[3] = np.array([w,h,0])

	return worldPnts

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









# ~~~~~~~~~~~~~~Start of MAAAAIIIINNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mtx = np.load("mtx.npy")
dist = np.load("dist.npy")

worldPoints = createRectWorldPoints(270,190)
sorted = sortCorners(worldPoints)
print(sorted)

while True:
	# grab the current frame and initialize the status text

	_, frame = camera.read()

	frame_to_show, detected, corners, last_aspectRatio = getCorners(frame, last_aspectRatio)

	if detected:
		corners = sortCorners(corners)
		print("Corners:\n",corners)
		pos = estimateCameraPose(corners,worldPoints,mtx,dist)
		print("Distance: ", np.linalg.norm(pos))

	cv2.imshow('Frame To Show',frame_to_show)

	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
