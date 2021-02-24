import cv2
import numpy as np

#Chapter 6: Joining Image

img = cv2.imread(r"D:/ML/New folder/Capture.PNG")
hor=np.hstack((img,img))
ver=np.vstack((img,img))
cv2.imshow()