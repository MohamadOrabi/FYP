import cv2
import numpy as np
from helpers.digit_recognition import DigitDetect
from helpers.corner_detection import *
from imutils.video import FPS

# Initialize DigitDetect Object
digitDetect = DigitDetect()
digitDetect.retrieve_model()
digit = None

mtx = np.load("data/mtx.npy")
dist = np.load("data/dist.npy")
last_aspectRatio = 0

worldPoints = createRectWorldPoints(297, 210, 263, 158, 17, 26)
sortCorners(worldPoints)

x_track, y_track, w_track, h_track = 0,0,0,0

camera = cv2.VideoCapture("Images/Vid.MOV")
#camera = cv2.VideoCapture(0)
fps = FPS().start()

try:
	while True:
		# grab the current frame and initialize the status text
		_, frame = camera.read()
		#f rame = cv2.imread('Image-50cm.jpeg')
		# h, w = frame.shape[:2]
		# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
		# frame = cv2.resize(frame, (1200, 1200))
		# frame = cv2.undistort(frame, newcameramtx, dist, None, mtx)

		frame = cv2.flip(frame, -1)

		frame_to_show, detected_corners, corners, moments, last_aspectRatio = getCorners(frame, last_aspectRatio)
		print("Detected Cornes:", detected_corners)
		print("len:", len(corners))

		if detected_corners and len(corners) >= 8:
			print("Corners Detected!")

			corners = checkCentroids(corners, moments)
			corners = sortCorners(corners)

			#Count the corner inside the image
			labelCorners(corners,frame_to_show)

			if len(corners) == 8:
				# Get Location of window to be tracked
				x_track = int(corners[4, 0])
				y_track = int(corners[4, 1])
				w_track = int(corners[6, 0] - x_track)
				h_track = int(corners[7, 1] - y_track)

				# Get digit in rectangle
				roi = frame[y_track:y_track + h_track, x_track:x_track + w_track]
				roi = cv2.bitwise_not(roi)
				digit = digitDetect.recognise_digit(roi)
				# cv2.imshow('roi', roi)
				print("Digit: ", digit)

				pos = estimateCameraPose(worldPoints, corners, mtx, dist)
				print("Distance: ", np.linalg.norm(pos))
			else:
				print(len(corners), " corners detected")
		else:
			print("Not Detected")

		cv2.imshow('Frame To Show',frame_to_show)
		#cv2.waitKey()
		if cv2.waitKey(1) & 0xFF == ord('q'):
		 	break

		fps.update()
except:
	print("Terminating Video")

fps.stop()
print("FPS: {:.2f}".format(fps.fps()))

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
