import cv2
import numpy as np
img=cv2.imread(r"D:/ML/OPENCV/penguin.jpg",1)
resized=cv2.resize(img,(int (img.shape[1]/2),int (img.shape[0]/2)))
cv2.imshow('image',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Chapter 1: Video Intro

cap=cv2.VideoCapture(0)#webcam
cap.set(3,640)
cap.set(4,48)
cap.set(10,0)

while True:
    success,img=cap.read()
    cv2.imshow("Video",img)

    kernel = np.ones((5, 5), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 150, 200)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

    cv2.imshow("Grey", imgGray)
    cv2.imshow("Blur", imgBlur)
    cv2.imshow("Canny", imgCanny)
    cv2.imshow("Dilation", imgDialation)
    cv2.imshow("Erode", imgEroded)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
