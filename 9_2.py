import cv2
import numpy as np

#Project 10:Virtual paint

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
