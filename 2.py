import cv2
import numpy as np

#Chapter 2:Basic Functions

img = cv2.imread(r"D:/ML/New folder/penguin.jpg")

img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

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
cv2.imshow("Erode",imgEroded)
cv2.waitKey(0)
