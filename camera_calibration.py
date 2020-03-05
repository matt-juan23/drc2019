#!/usr/bin/env python3

import cv2
import numpy as np

def nothing(x):
    pass
    
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
WIDTH = int(cap0.get(3) * 2)
HEIGHT = int(cap0.get(4))

while (1):
    # Take each frame
    _, frame0 = cap0.read()
    _, frame1 = cap1.read()
    frame = np.concatenate((frame1, frame0), axis=1)
    # crop top area of image
    cframe = frame[int(HEIGHT/2)-180:HEIGHT, 0:WIDTH]
    
    #cv2.imshow('frame', frame)
    cv2.imshow('cframe', cframe)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap0.release()
cv2.destroyAllWindows()
