import cv2
import numpy as np
#Chapter 9: Face Detection

facecascade=cv2.CascadeClassifier("D:/ML/New folder/3 hours course/opencv-master/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")

img=cv2.imread("D:/ML/New folder/Starc56.png")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=facecascade.detectMultiScale(img,1.1,4)
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("Result")
cv2.waitKey(0)


