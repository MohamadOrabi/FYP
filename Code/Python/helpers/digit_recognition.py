import cv2
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
	# def create_model(self):
	# 	(self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
	# 	# Reshaping the array to 4-dims so that it can work with the Keras API
	# 	self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
	# 	self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
	# 	# Making sure that the values are float so that we can get decimal points after division
	# 	self.x_train = self.x_train.astype('float32')
	# 	self.x_test = self.x_test.astype('float32')
	# 	# Normalizing the RGB codes by dividing it to the max RGB value.
	# 	self.x_train /= 255
	# 	self.x_test /= 255
	# 	# Creating a Sequential Model and adding the layers
	# 	self.model = Sequential()
	# 	self.model.add(Conv2D(28, kernel_size=(3, 3), input_shape=DigitDetect.input_shape))
	# 	self.model.add(MaxPooling2D(pool_size=(2, 2)))
	# 	self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
	# 	self.model.add(Dense(128, activation='relu'))
	# 	self.model.add(Dropout(0.2))
	# 	self.model.add(Dense(10, activation='softmax'))

	# Train model
	# def train_model(self):
	# 	self.model.compile(optimizer='adam',
	# 					   loss='sparse_categorical_crossentropy',
	# 					   metrics=['accuracy'])
	# 	self.model.fit(x=self.x_train, y=self.y_train, epochs=10)

	# # Test model
	# def test_model(self):
	# 	loss, acc = self.model.evaluate(self.x_test, self.y_test)
	# 	return loss, acc

	# # Save trained model
	# def save_model(self):
	# 	self.model.save('data/final_model.h5')

	# Retrieve saved model
	def retrieve_model(self):
		self.model = load_model('data/final_model.h5')

	# Takes as input the image cropped with only the digit in it
	def recognise_digit(self, img):
		# load the image
		# img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (28, 28))
		# convert to array
		img = img_to_array(img)
		# reshape into a single sample with 1 channel
		img = img.reshape(1, 28, 28, 1)
		# prepare pixel data
		img = img.astype('float32')
		img = img / 255.0
		digit = self.model.predict_classes(img)
		return digit[0]
