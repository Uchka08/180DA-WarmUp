#Reference https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

import cv2 as cv
import numpy as np


#capture from FaceTime camera
cap = cv.VideoCapture(0)

while True:
    # if frame reads correctly, good returns True
    # frame is an numpy array of arrays
    good, frame = cap.read()
    
    # Convert BGR to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange (hsv, lower_blue, upper_blue)
    cnt = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    # For bounding box
    if len(cnt)>0:
        blue_area = max(cnt, key=cv.contourArea)
        (x,y,w,h) = cv.boundingRect(blue_area)
        cv.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)

    cv.imshow('frame',frame)

    k = cv.waitKey(5) 
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()