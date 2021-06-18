import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from PIL import Image

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_righteye_2splits.xml')


import tensorflow as tf
model = tf.keras.models.load_model("./openorclose2.h5")

lbl=['Close','Open']

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    right_eye_check,left_eye_check=0,0
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        cv2.imwrite('rightEye.jpg',r_eye)
        img=cv2.imread('./rightEye.jpg')
        img= Image.fromarray(img)
        img = img.resize((150, 150))
        input_img = np.expand_dims(np.array(img), axis=0) 
        right_eye_check=round((model.predict(input_img)[0][0]))
        
        if(right_eye_check==0):
            lbl='Open' 
        if(right_eye_check==1):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        cv2.imwrite('leftEye.jpg',l_eye)
        img=cv2.imread('./leftEye.jpg')
        img= Image.fromarray(img)
        img = img.resize((150,150))
        input_img = np.expand_dims(np.array(img), axis=0) 
        left_eye_check=round((model.predict(input_img)[0][0]))
        
        if(left_eye_check==0):
            lbl='Open' 
        if(left_eye_check==1):
            lbl='Closed'
        break
    #print(right_eye_check," ",left_eye_check)
    if(right_eye_check==1 and left_eye_check==1):
        if score < 10:
            score=score+1

        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score >= 10):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite('image.jpg',frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
