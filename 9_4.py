import cv2
import numpy as np

#Project 2: Document Scanner

widthImg=800
heightImg=300


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


#Preprocessing
def preprocessing(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgblur,200,200)
    kernel=np.ones((5,5))
    imgdial=cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres=cv2.erode(imgdial,kernel,iterations=1)
#    cv2.imshow("imgt",imgThres)
    return imgThres

def getcounters(img):
    biggest=np.array([])
    maxarea=0
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            if area>maxarea and len(approx)==4:
                biggest=approx
                maxarea=area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),20)
#    cv2.imshow("imgcon",imgContour)
    print(biggest)
    return biggest

def getWarp(img,biggest):
    biggest=reorder(biggest)
    pts1=np.float32(biggest)
    pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    imgCropped=imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped=cv2.resize(imgCropped,(widthImg,heightImg))
    return imgCropped

def reorder(mypoints):
    mypoints=mypoints.reshape((4,2))
    mypointsnew=np.zeros((4,1,2),np.int32)
    add=mypoints.sum(1)
    print("add",add)
    mypointsnew[0]=mypoints[np.argmin(add)]
    mypointsnew[3]=mypoints[np.argmax(add)]
    diff=np.diff(mypoints,axis=1)
    mypointsnew[1]=mypoints[np.argmin(diff)]
    mypointsnew[2]=mypoints[np.argmax(diff)]
    return mypointsnew

cap=cv2.VideoCapture(0)
cap.set(10,150)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgThres = preprocessing(img)
    biggest = getcounters(imgThres)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)           # imageArray = ([img,imgThres],
        imageArray = ([imgContour, imgWarped])       #         [imgContour,imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imageArray = ([imgContour, img])                                     # imageArray = ([img, imgThres],
                                                                             #               [img, img])
    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


