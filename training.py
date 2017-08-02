import sys

import numpy as np
import cv2

im = cv2.imread('training.jpg')
im3 = im.copy()

roii = im[2:,59:1033,:]
gray = cv2.cvtColor(roii,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if 100 < cv2.arcLength(cnt,True) < 250:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  (h < 60) & (w < 60) :
            cv2.drawContours(roii,[cnt],0,(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('trainingsamples.data',samples)
np.savetxt('trainingresponses.data',responses)
