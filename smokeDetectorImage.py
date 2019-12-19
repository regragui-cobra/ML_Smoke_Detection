import numpy as np
import cv2

smoke_cascade  = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('test.jpg')
#imgresize = cv2.resize(img,(100,100), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

smoke = smoke_cascade.detectMultiScale(gray,1.01,7)
print(smoke)
for(x,y,w,h) in smoke:
    imgresize = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
