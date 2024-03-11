import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
# Load the trained model
model = load_model('hand_gesture_recognition_model.h5')
imgSize=256
offset=20
labels=['peace','rock','stop','thumbs_down','thumbs_up']
# folder="Data/rock"
# count=0
while(True):
    success,img =cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
    if(hands):
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]

        aspectRatio=h/w
        if(aspectRatio>1):
            k=imgSize/h
            wcal=math.ceil(k*w)
            if(imgCrop.size):
                imgResize=cv2.resize(imgCrop,(wcal,imgSize))
                imgResizeShape=imgResize.shape
                wgap=math.ceil((imgSize-wcal)/2)
                imgWhite[:,wgap:wcal+wgap]=imgResize
                imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
                imgWhite = imgWhite / 255.0
                predictions = model.predict(imgWhite)
                # Get the predicted class label
                predicted_label = labels[np.argmax(predictions)]
                print(predicted_label)
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            if (imgCrop.size):
                imgResize = cv2.resize(imgCrop, (imgSize,hcal))
                imgResizeShape = imgResize.shape
                hgap = math.ceil((imgSize - hcal) / 2)
                imgWhite[hgap:hcal + hgap, :] = imgResize
                imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
                imgWhite = imgWhite / 255.0
                predictions = model.predict(imgWhite)
                # Get the predicted class label
                predicted_label = labels[np.argmax(predictions)]
                print(predicted_label)
        cv2.putText(imgOutput, predicted_label,(60,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        # cv2.imshow('whiteImage', imgWhite)
        # print(imgWhite.shape)
    cv2.imshow('Camera',imgOutput)

    cv2.waitKey(1)
    # if key==ord("s"):
    #     cv2.imwrite(f'{folder}/Image{count}.jpg', imgWhite)
    #     count+=1
    #     print(count)