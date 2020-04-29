import cv2
import numpy as np
import requests
from helpers.digit_recognition import DigitDetect
from helpers.corner_detection import *
import socket

# Initialize DigitDetect Object
digitDetect = DigitDetect()
digitDetect.retrieve_model()
digit = None

mtx = np.load("data/mtx.npy")
dist = np.load("data/dist.npy")
last_aspectRatio = 0

worldPoints = createRectWorldPoints(w1=413, h1=280, w2=340, h2=205, dw=38, dh=38)
sortCorners(worldPoints)

x_track, y_track, w_track, h_track = 0,0,0,0

url = "http://admin:admin@192.168.43.1:8080/video"

#camera = cv2.VideoCapture("Images/Vid.MOV")
camera = cv2.VideoCapture(url)
#camera = cv2.VideoCapture(0)

# while True:
# 	_, frame = camera.read()
# 	cv2.imshow('frame', frame)
# 	cv2.waitKey(1)


HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))

	pos = 0
	while True:
		# grab the current frame and initialize the status text
		_, frame = camera.read()
		#frame = cv2.flip(frame, -1)

		#frame = cv2.imread('Images/Im5.jpeg')
		# h, w = frame.shape[:2]
		# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
		# frame = cv2.resize(frame, (1200, 1200))
		# frame = cv2.undistort(frame, newcameramtx, dist, None, mtx)

		#detectVanishingPoints(frame)
		#frame = blobDetection(frame)

		corners, moments = getCorners(frame)

		if len(corners) >= 8:
			#print("Corners Detected!")

			corners = checkCentroids(corners, moments, thresh=80, vthresh=300, cthresh=10, frame=frame)
			corners = sortCorners(corners)

			labelCorners(corners, frame)

			if len(corners) == 8:
				# Get Location of window to be tracked
				x_track = int(corners[4, 0])
				y_track = int(corners[4, 1])
				w_track = int(corners[6, 0] - x_track)
				h_track = int(corners[7, 1] - y_track)

				# Get digit in rectangle
				if h_track >= 5 and w_track >= 5:
					roi = frame[y_track:y_track + h_track, x_track:x_track + w_track]
					roi = cv2.bitwise_not(roi)
					digit = digitDetect.recognise_digit(roi)

					if digit is not None:
						pos = estimateCameraPose(worldPoints, corners, mtx, dist)
					
						#The string to be sent should be in the format: label + white space + x + white space + y
						#For example when retreiving shampoo: 2 0 1.3
						#str1 = str(digit) + " {:.2f} {:.2f}".format(pos[0]/1000, -pos[2]/1000)
						str1 = "2 {:.2f} {:.2f}".format(pos[0]/1000, -pos[2]/1000)
						print(str1, "WAS SENT TO SERVER")
						s.sendall(str1.encode())
						print("Digit: ", digit, "Distance: ", np.linalg.norm(pos))


		frame = cv2.resize(frame, (800, 600))
		cv2.putText(frame, "Distance: " + str(np.linalg.norm(pos)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.imshow('Frame To Show', frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()