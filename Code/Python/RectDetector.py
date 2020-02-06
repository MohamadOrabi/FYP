import numpy as np
import cv2 as cv

img = cv.imread('./Images/im2.jpeg')
img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Original',img)
#cv.imshow('Grayscale',img_gray)
cv.waitKey(1)


#ret,thresh = cv.threshold(img,127,255,0)
#cv.imshow('Threshold',img)
#cv.waitKey(1)

lower = np.array([0,0,100])  #-- Lower range --
upper = np.array([255,255,255])  #-- Upper range --
mask = cv.inRange(img_HSV, lower, upper)

contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
image = cv.drawContours(img, contours, -1, (0, 255, 0), 2)


cv.imshow('Contours', image)
cv.imshow('Mask',mask)



cv.waitKey()

#im2,contours,hierarchy = cv.findContours(thresh, 1, 2)


# cap = cv.VideoCapture(0)

# while True:
# 	_, img = cap.read()
# 	cv.imshow('img', img)
# 	if cv.waitKey(1) & 0xff == ord('q'):
# 		break
