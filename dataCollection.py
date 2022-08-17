import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import  Classifier
import math
import time

classifier = Classifier('Model/model2.h5','Model/labels.txt')
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

while True:
    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands :
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            wOffset = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wOffset:wCal+wOffset] = imgResize
            pred,index = classifier.getPrediction(imgWhite)
            print(pred,index)
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            hOffset = math.ceil((imgSize - hCal) / 2)
            imgWhite[hOffset:hCal+hOffset,:] = imgResize
            pred, index = classifier.getPrediction(imgWhite)
            print(pred, index)
        cv2.imshow("imageWhite", imgWhite)
        cv2.imshow("imagecrop", imgCrop)
    cv2.imshow("image",img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
