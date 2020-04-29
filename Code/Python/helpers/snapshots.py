import cv2
import requests
import numpy as np

url = "http://admin:admin@192.168.43.1:8080/video"

for i in range(0, 20):
    camera = cv2.VideoCapture(url)
    for x in range(100000000):
        continue
    _, img = camera.read() 
    cv2.imwrite('../Images/server/img' + str(i) + '.jpeg', img)
    print('saved')