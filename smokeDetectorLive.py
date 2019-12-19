
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2
#initialisation du Haar Classifier Cascade
cascade_src ='cascade.xml'
smoke_cascade=cv2.CascadeClassifier(cascade_src)

# initialisation des paramètres pour la capture
camera = PiCamera()
camera.resolution = (800, 600)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(800, 600))

# temps réservé pour l'autofocus
time.sleep(0.1)

# capture du flux vidéo
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

# recupère à l'aide de Numpy le cadre de l'image, pour l'afficher ensuite à l'écran
             image = frame.array
             
# detection de la fumée

             gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

             smoke = smoke_cascade.detectMultiScale(gray,1.1,1)

             for(x,y,w,h) in smoke:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

# affichage du flux vidéo
             cv2.imshow("Video", image)
             key = cv2.waitKey(1) & 0xFF

# initialisation du flux 
             rawCapture.truncate(0)

# si la touche q du clavier est appuyée, on sort de la boucle
             if key == ord("q"):
                        break
