#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:15:40 2018

@author: neel
"""

# Loading the libraries
import cv2

#Loading the Cascades(cascade is serious of filters to detect faces one after the other)
#Cascades only work on blac-white images(gray) and original image(frame)
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

#Defining functions to do the detections
# Functions would work on single images only not videos
def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #image will be reduced 1.3 times
    #in order to zone of pixel to accepted 5 of the nearest zones should be accepted in webcam
    for(x,y,w,h) in faces:
        cv2.rectange(frame,(x,y),(x+w,y+h),(255,0,0),2) 
        # Created rectangles around faces and now its time to draw eye
        roi_gray= gray[y:y+h,x:x+w]
        roi_color= frame[y:y+h,x:x+w]
        # now we have to zones of intrest one for frame and one for original image
        # we drew rectangels surrounding the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,22)
        #Time to create new for loop to draw rectangles around the eyes like I did for faces
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        smile = smile_cascade.detectMultiScale(roi_gray,1.7,22)
        #Selected larger scaling factor from 1.1 to 1.7 and increased neighbour count to get 
        #one rectangle detector while smilling
        #Time to create new for loop to draw rectangles around the eyes like I did for faces
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)    
    return frame       
            #(0,255,0) for green colour
            
# Doing Some Face recognition with webcam 
video_capture=cv2.VideoCapture(0) 
#0 for internal webcam, 1 for external webcam 
while True:
    _, frame=video_capture.read()
    #Convert frame to gray
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #result of the detect function will stored in canvas
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    #Stop webcam once done
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
# Turn off the webcam and destroy images:-
video_capture.release()
cv2.destroyAllWindows()  