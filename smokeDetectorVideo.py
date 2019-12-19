import numpy as np
import cv2

cascade_src ='cascade.xml'
video_src ='videotest.mp4'
cap=cv2.VideoCapture(video_src)
smoke_cascade=cv2.CascadeClassifier(cascade_src)
while True:
    ret,img=cap.read()

    if(type(img) == type(None)):
        break

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    smoke = smoke_cascade.detectMultiScale(gray,1.1,1)

    for(x,y,w,h) in smoke:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('video',img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
