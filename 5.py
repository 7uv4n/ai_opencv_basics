import cv2
import numpy as np

#Chapter 5: Warp Perspective

img = cv2.imread(r"D:/ML/New folder/Capture.PNG")
width,height=250,350
pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgOut=cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("first",img)
cv2.imshow("first",imgOut)
cv2.waitKey(0)
cv2.destroyAllWindows()
