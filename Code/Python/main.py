#import helpers
import imutils
import cv2
import numpy as np
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# Digit Detection
class DigitDetect:
	input_shape = (28, 28, 1)

	# Empty constructor
	def __init__(self):
		pass

	# Create new model
	def create_model(self):
		(self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
		# Reshaping the array to 4-dims so that it can work with the Keras API
		self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
		self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
		# Making sure that the values are float so that we can get decimal points after division
		self.x_train = self.x_train.astype('float32')
		self.x_test = self.x_test.astype('float32')
		# Normalizing the RGB codes by dividing it to the max RGB value.
		self.x_train /= 255
		self.x_test /= 255
		# Creating a Sequential Model and adding the layers
		self.model = Sequential()
		self.model.add(Conv2D(28, kernel_size=(3, 3), input_shape=DigitDetect.input_shape))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(10, activation='softmax'))

	# Train model
	def train_model(self):
		self.model.compile(optimizer='adam',
						   loss='sparse_categorical_crossentropy',
						   metrics=['accuracy'])
		self.model.fit(x=self.x_train, y=self.y_train, epochs=10)

	# Test model
	def test_model(self):
		loss, acc = self.model.evaluate(self.x_test, self.y_test)
		return loss, acc

	# Save trained model
	def save_model(self):
		self.model.save('final_model.h5')

	# Retrieve saved model
	def retrieve_model(self):
		self.model = load_model('final_model.h5')

	# Takes as input the image cropped with only the digit in it
	def recognise_digit(self, img):
		# load the image
		# img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (28, 28))
		# convert to array
		# img = img_to_array(img)
		# reshape into a single sample with 1 channel
		img = img.reshape(1, 28, 28, 1)
		# prepare pixel data
		img = img.astype('float32')
		img = img / 255.0
		digit = self.model.predict_classes(img)
		return digit[0]

def cornerIn(curr, image_points):
	for x, y in image_points:
		if abs(x - curr[0]) < 25 and abs(y - curr[1]) < 25:
			return True
	return False

def getCorners(frame,last_aspectRatio):
	corners = np.empty([0, 2], dtype=np.float32)
	current_corners = np.empty([0, 2], dtype=np.float32)

	detected_corners = False
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
				detected_corners = True
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
	return frame, detected_corners, corners, last_aspectRatio

def estimateCameraPose(objp, corners, mtx, dist):

	pos = 0

	try:
		ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
		tvec = np.squeeze(tvec)

		if ret:
			Rt,jacob = cv2.Rodrigues(rvec)
			R = Rt.transpose()
			#print("R",Rt)
			pos = tvec.dot(-Rt)
			print("Distance: ",np.linalg.norm(pos))
			print("pos",pos)
	except:
		warnings.warn("Error is estimateCameraPose")
		print("\n objp:", objp,
			  "\n corners:", corners,
			  "\n mtx:", mtx,
			  "\n dst:", dist,)

	return pos

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

def trackRect(frame,x,y,w,h):	#We can probably change this function to pass roi directly
	track_window = (x, y, w, h)
	roi = frame[y:y + h, x:x + w]
	hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255., 255., 255.)))
	roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)	#This could probably be calculated somewhere else
	dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

	# apply CamShift to get the new location
	ret, track_window = cv2.CamShift(dst, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1))

	# Draw it on image
	x, y, w, h = track_window
	img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
	cv2.imshow('img2', img2)
	return x,y,w,h

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

# Initialize DigitDetect Object
digitDetect = DigitDetect()
# Commented training the model since it is already trained and saved
# digitDetect.create_model()q
# digitDetect.train_model()
# loss, acc = digitDetect.test_model()
# print("Loss: %.3f, Accuracy: %.3f" %(loss,acc))
# digitDetect.save_model()
digitDetect.retrieve_model()

digit = None

mtx = np.load("mtx.npy")
dist = np.load("dist.npy")
last_aspectRatio = 0

worldPoints = createRectWorldPoints(22, 250, 170, 207, 123)
sortCorners(worldPoints)
print("World Points:\n", worldPoints)

first_detection = True
x_track, y_track, w_track, h_track = 0,0,0,0

camera = cv2.VideoCapture("Images/Vid.mov")

while True:
	# grab the current frame and initialize the status text
	_, frame = camera.read()
	frame = cv2.flip(frame, -1)

	if first_detection:
		frame_to_show, detected_corners, corners, last_aspectRatio = getCorners(frame, last_aspectRatio)
		if detected_corners:
			first_detection = True

			#Get Location of window to be tracked
			x_track = int(np.min(corners[:,0]))
			y_track = int(np.min(corners[:,1]))
			w_track = int(np.max(corners[:,0]) - np.min(corners[:,0]))
			h_track = int(np.max(corners[:,1]) - np.min(corners[:,1]))

			#Get digit in rectangle
			roi = frame[y_track:y_track + h_track, x_track:x_track + w_track]
			roi = cv2.bitwise_not(roi)
			digit = digitDetect.recognise_digit(roi)
			print("Digit: ", digit)

	else:
		#frame_to_track = frame[y_track:y_track+h_track, x_track:x_track+w_track]
		#cv2.imshow('frame',frame)
		frame_to_show, detected_corners, corners, last_aspectRatio = getCorners(frame, last_aspectRatio)
		#x_track,y_track,w_track,h_track = trackRect(frame,x_track,y_track,w_track,h_track)

	if detected_corners:
		print("Detected!")
		corners = sortCorners(corners)
		print("Corners:\n", corners)

		#Count the corner inside the image
		counter = 0
		for x, y in corners:
			counter+=1
			cv2.circle(frame_to_show, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=8, shift=0)
			cv2.putText(frame_to_show, str(counter), (int(x + 12), int(y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

		pos = estimateCameraPose(worldPoints, corners, mtx, dist)
		print("Distance: ", np.linalg.norm(pos))
	else:
		print("Not Detected")

	cv2.imshow('Frame To Show',frame_to_show)
	cv2.waitKey()

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
