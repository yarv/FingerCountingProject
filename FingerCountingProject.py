from importlib.resources import path
import cv2
import time
import os
import HandTrackingModule as htm
from google.protobuf.json_format import MessageToDict

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"

pathList = os.listdir(folderPath)
pathList = sorted(pathList)
overlayList = []

for imPath in pathList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)



prev_time = time.time()
cur_time = time.time()

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while (cur_time - prev_time) < 10:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        handedness = MessageToDict(detector.results.multi_handedness[0])
        whichHand = handedness['classification'][0]['label']
        fingers = []

        # Thumb:

        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            if whichHand == 'Right':
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if whichHand == 'Right':
                fingers.append(0)
            else:
                fingers.append(1)

        # rest of fingers:
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = sum(fingers)

        overlay = overlayList[totalFingers-1]
        overlayH, overlayW, overlayC = overlay.shape
        img[0:overlayH, 0:overlayW] = overlay
    

    cur_time = time.time()

    fps = 1/(cur_time-prev_time)
    prev_time = cur_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    