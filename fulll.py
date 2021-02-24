import cv2
import numpy as np
"""
Chapter 1: Image Intro

img=cv2.imread(r"D:/ML/OPENCV/penguin.jpg",1)
resized=cv2.resize(img,(int (img.shape[1]/2),int (img.shape[0]/2)))
cv2.imshow('image',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

Chapter 1: Video Intro
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

Chapter 2:Basic Functions

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

Chapter 3: Resizing and cropping



img = cv2.imread(r"D:/ML/New folder/penguin.jpg")
img1=img[0:1200,200:1300]
cv2.imshow("first",img)
cv2.imshow("Second",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

Chapter 4: Shapes and Texts

img=np.zeros((512,512,3),np.uint8)
img[:]=0,0,0
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),5)
cv2.rectangle(img,(0,0),(int(img.shape[1]/2),int(img.shape[0]/2)),(0,0,255),cv2.FILLED)
cv2.circle(img,(400,50),30,(255,255,0),cv2.FILLED)
cv2.putText(img,"OPENCV",(300,300),cv2.FONT_HERSHEY_COMPLEX,1,(150,0,0),1)
cv2.imshow("Second",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Chapter 5: Warp Perspective
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

Chapter 6: Joining Image

img = cv2.imread(r"D:/ML/New folder/Capture.PNG")
hor=np.hstack((img,img))
ver=np.vstack((img,img))
cv2.imshow()

Chapter 7: Color Detection

def empty(a):
    pass


img = cv2.imread(r"D:/ML/New folder/penguin.jpg")
img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Hue Min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue Max","Trackbars",179,179,empty)
cv2.createTrackbar("Sat Min","Trackbars",0,255,empty)
cv2.createTrackbar("Sat Max","Trackbars",255,255,empty)
cv2.createTrackbar("Val Min","Trackbars",0,255,empty)
cv2.createTrackbar("Val Max","Trackbars",255,255,empty)

imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    h_min=cv2.getTrackbarPos("Hue Min","Trackbars")
    h_max=cv2.getTrackbarPos("Hue Max","Trackbars")
    s_min=cv2.getTrackbarPos("Sat Min","Trackbars")
    s_max=cv2.getTrackbarPos("Sat Max","Trackbars")
    v_min=cv2.getTrackbarPos("Val Min","Trackbars")
    v_max=cv2.getTrackbarPos("Val Max","Trackbars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHsv,lower,upper)
    imgResult=cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Orginal",img)
    cv2.imshow("Hsv",imgHsv)
    cv2.imshow("Mask",mask)
    cv2.imshow("Result",imgResult)
    cv2.waitKey(1)

Chapter 8: Contours and Shape Detections


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

def getcontours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgcontor,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            print(peri)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objcor=len(approx)
            x, y, w, h= cv2.boundingRect(approx)
            if objcor==3:objType="Tri"
            elif objcor==4:
                asp=w/h
                if asp>0.95 and asp<1.0: objType="Squarr"
                else: objType="Rectangle"
            elif objcor>4:objType="Circle"
            else:objType="NONE"

            cv2.rectangle(imgcontor,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgcontor,objType,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)

img = cv2.imread(r"D:/ML/New folder/shapes.PNG")
imgcontor=img.copy()


imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)
imgBlank=np.zeros_like(img)
getcontours(imgCanny)


imgStack=stackImages(0.6,([img,imgGray,imgBlur],[imgCanny,imgcontor,imgBlank]))


cv2.imshow("Original",imgStack)
cv2.waitKey(0)


Chapter 9: Face Detection

facecascade=cv2.CascadeClassifier("D:/ML/New folder/3 hours course/opencv-master/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")

img=cv2.imread("D:/ML/New folder/Starc56.png")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=facecascade.detectMultiScale(img,1.1,4)
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("Result")
cv2.waitKey(0)


Chapter 9: Face Detection (Webcam)
facecascade=cv2.CascadeClassifier("D:/ML/New folder/3 hours course/opencv-master/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)

while True:
    scale,img=cap.read()
    faces=facecascade.detectMultiScale(img,1.1,4)
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"U1",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

    cv2.imshow("Result",img)
    cv2.waitKey(1)


Project 10:Virtual paint

#done with one color
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,48)
cap.set(10,0)

myPoints=[]
mycolors=[[30,117,40,113,160,255]]
mycolorvalues=[[255,255,255]]

def getcontours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h=0,0,0,0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #print(area)
        if area>500:
            #cv2.drawContours(imresult,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            #print(peri)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            #objcor=len(approx)
            x, y, w, h= cv2.boundingRect(approx)
    return x,y

def findColor(img,mycolors,mycolorvalues):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    newpoints=[]
    lower=np.array(mycolors[0][0:3])
    upper=np.array(mycolors[0][3:6])
    mask=cv2.inRange(img,lower,upper)
    x,y=getcontours(mask)
    cv2.circle(imresult,(x,y),10,mycolorvalues[0],cv2.FILLED)
    if x!=0 and y!=0:
        newpoints.append([x,y])
    #cv2.imshow("img",mask)
    return newpoints

def draw(mypoints,mycolor):
  for point in myPoints:
      cv2.circle(imresult, (point[0], point[1]), 10, (255,255,255), cv2.FILLED)

while True:
    success,img=cap.read()
    imresult=img.copy()
    newpoints=findColor(img,mycolors,mycolorvalues)
    if len(newpoints)!=0:
        for newp in newpoints:
            myPoints.append(newp)
    print(myPoints)
    if len(myPoints)!=0:
        draw(myPoints,mycolorvalues)
    cv2.imshow("Video",imresult)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

        
Project side:HSV in WEBCAM

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

def empty(a):
    pass

cap=cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Hue Min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue Max","Trackbars",179,179,empty)
cv2.createTrackbar("Sat Min","Trackbars",0,255,empty)
cv2.createTrackbar("Sat Max","Trackbars",255,255,empty)
cv2.createTrackbar("Val Min","Trackbars",0,255,empty)
cv2.createTrackbar("Val Max","Trackbars",255,255,empty)

while True:
    success,img=cap.read()

    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min=cv2.getTrackbarPos("Hue Min","Trackbars")
    h_max=cv2.getTrackbarPos("Hue Max","Trackbars")
    s_min=cv2.getTrackbarPos("Sat Min","Trackbars")
    s_max=cv2.getTrackbarPos("Sat Max","Trackbars")
    v_min=cv2.getTrackbarPos("Val Min","Trackbars")
    v_max=cv2.getTrackbarPos("Val Max","Trackbars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHsv,lower,upper)
    imgResult=cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("Stack",stackImages(0.6,([img,imgHsv,mask,imgResult])))
    cv2.imshow("Orginal",img)
    cv2.imshow("Hsv",imgHsv)
    
    30 117 40 113 160 255
    cv2.imshow("Mask",mask)
    cv2.imshow("Result",imgResult)
        cv2.waitKey(1)

Project 2: Document Scanner
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


"""
