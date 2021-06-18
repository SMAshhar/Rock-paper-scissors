from typing import no_type_check
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence = 0.7, min_tracking_confidence=0.7, max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
ret, img = cap.read()
h, w, c = img.shape
# temp = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# data_frame = pd.DataFrame(temp, columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
x = np.zeros((1, 40))
data = pd.DataFrame(x)
with hands:
    while (cv2.waitKey(1)-27):
        ret, frame = cap.read()
        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgrgb.flags.writeable = False
        results = hands.process(imgrgb)
        
        xlst = []
        ylst = []
        iD=[]
        bbox = []
        coord=[]
        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                for ids, lm in enumerate(handlm.landmark): # for bounding box
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print('ID:',ids,'X:',cx,'Y:',cy)
                    iD.append(ids)
                    xlst.append(cx)
                    xlst.append(cy)
                    
                    ylst.append(cy)
                data = pd.concat([data,pd.DataFrame(xlst).T], axis=0)
                print(data)
                
                # for i in range(len(iD)):
                #     coord.append((xlst[i],ylst[i]))
                

                mpDraw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(color = (0,0,255),thickness = 1, circle_radius=3),mpDraw.DrawingSpec(color = (255,0,0),thickness = 1, circle_radius=3),)
                xmin, xmax = min(xlst), max(xlst)
                ymin, ymax = min(ylst), max(ylst)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),(bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),(0, 255, 0), 2)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fpss = str('FPS:'+str(int(fps)))
        data.to_csv('game.csv')
        cv2.putText(frame, fpss, (w-112,h-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),1)
        cv2.imshow('frame',frame)
        if len(data) >= 1000:
            break
