from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import random
import sys
import os

uncodedPath = "/home/username/some/input/dir"
codedPath = "/home/username/some/output/dir"
keypoints = {'Point_A': (100,100),'Point_B': (100,200), 'Point_C': (100,300),'Point_D': (100,400)}

print("use the mouse to drag the keypoints")
print("enter: save")
print("space: skip")
print("entf: delete")
print("esc: close")


def getDistance(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    
def getFilesInDir(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def gradientMap(image):
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray,(51,51),3)
    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0));
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1));
    gray=cv2.addWeighted(grad_x, 0.5,grad_y, 0.5, 0);
    cv2.normalize(gray,gray,180,0,cv2.NORM_MINMAX)
    hsv = np.ones((gray.shape[0],gray.shape[1],3),dtype='uint8')
    hsv*=255
    hsv[:,:,0]=gray
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  



mouseIsDown = False
mousePixelPos = (0,0)
def mouse(event, x, y, flags, param):
    global mouseIsDown,mousePixelPos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseIsDown = True
    if event == cv2.EVENT_LBUTTONUP:
        mouseIsDown = False
    if event == cv2.EVENT_MOUSEMOVE:
        mousePixelPos = (x,y)
        

uncodedFiles = getFilesInDir(uncodedPath)
codedFiles = getFilesInDir(codedPath)

todoFiles = []
for f in uncodedFiles:
    if f not in codedFiles:
        todoFiles.append(f)

random.shuffle(todoFiles)
print(str(len(todoFiles))+" files to do")


cv2.imshow('display',False)
cv2.setMouseCallback("display", mouse)

for f in todoFiles:
    img = cv2.imread(join(uncodedPath,f))
    gradMap = gradientMap(img)
    print("editing "+f)

    while True:
        dispaly = img.copy()


        #move keypoints using the mouse
        if mouseIsDown:
            #get nearest keypoint
            dispaly=gradMap.copy()
            keypointToMove = False
            distance = 50
            for keypoint in keypoints:
                d = getDistance(keypoints[keypoint],mousePixelPos)
                if d<distance:
                    distance = d
                    keypointToMove = keypoint
            if keypointToMove:
                keypoints[keypointToMove] = mousePixelPos

        #draw keypoints on the display
        for keypoint in keypoints:
            cv2.circle(dispaly, keypoints[keypoint], 5, (128, 128, 128),2)
            cv2.putText(dispaly, keypoint, keypoints[keypoint], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128))
        cv2.imshow('display',dispaly)
            
        #get keyboard input
        key = cv2.waitKey(1)
        if key!=-1:
            #print(key)
            if key == 13: #enter -> save coded sample
                print("saving " + f)
                cv2.imwrite(join(codedPath,f),img)
                file = open(join(codedPath,f+".csv"),"w")
                for keypoint in keypoints:
                    file.write(keypoint + ":" + str(keypoints[keypoint][0]) + "," + str(keypoints[keypoint][1])+"\n") 
                file.close() 
                break
            if key == 32: #space -> skip sample
                print("skipping " + f)
                break
            if key == 255: #entf -> delete sample from (uncoded) dataset
                print("deleting " + f)
                os.remove(join(uncodedPath,f))
                break
            if key == 27: #esc -> close program
                print("closing program")
                sys.exit(0)
                
                

