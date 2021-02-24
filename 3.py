import cv2
import numpy as np

#Chapter 3: Resizing and cropping



img = cv2.imread(r"D:/ML/New folder/penguin.jpg")
img1=img[0:1200,200:1300]
cv2.imshow("first",img)
cv2.imshow("Second",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
