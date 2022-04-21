import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import time
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('No se puede abrir la cámara')

Ts = 0.5 #Tiempo de muestreo
datat = ['Time (s)']
dataE = ['Emotion']

t0 = time.time() #Tiempo de inicio
while True:
    time.sleep(Ts)
    try:
        ret,frame = cap.read() #Lee una imagen del video
        result = DeepFace.analyze(frame,actions=['emotion'])

        ##############################################
        #Hace el rectangulo alrededor de la cara
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)

        ##############################################
        ##############################################

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, 
                    result['dominant_emotion'],
                   (150,150), font, 3, (0,0,255), 2, cv2.LINE_4);
        cv2.imshow('Demo video', frame)
        #Se guardan los datos
        datat.append(round( time.time()-t0, 2))
        dataE.append(result['dominant_emotion'])

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    except (ValueError) as err:
        #print(err)
        pass
        
cap.release()
cv2.destroyAllWindows()

nt=np.array(datat)
nE=np.array(dataE)
np.savetxt('Data.csv', [nt,nE], delimiter=';', fmt='%s')
    
