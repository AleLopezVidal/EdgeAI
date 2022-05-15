import cv2
#from keras.models import load_model
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import image
import matplotlib.pyplot as plt
import time
import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = tflite.Interpreter(model_path = 'emotion.tflite')
emotion_model.allocate_tensors()

#Obtener inputs y outputs de los tensores
emotion_input = emotion_model.get_input_details()
emotion_output = emotion_model.get_output_details()

emotion_shape = emotion_input[0]['shape']

emotion_classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('No se puede abrir la cámara')

Ts = 0.5 #Tiempo de muestreo

datat = ['Time (s)'] #Se guarda el dato del tiempo
dataE = ['Emotion'] #Se guarda el dato de la emoción
#data = ['Time (s)','Emotion']
t0 = time.time() #Tiempo de inicio, para tener referencia

while True:
    time.sleep(Ts)
    try:
        ret,frame = cap.read() #Lee una imagen del video
        #result = DeepFace.analyze(frame,actions=['emotion'])

        ##############################################
        #Hace el rectangulo alrededor de la cara
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)

        #Aquí se detecta la cara y sus facciones
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)

            #Se procesa la imagen para la predicción
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            #Se utiliza la imagen procesada en el modelo
            emotion_model.set_tensor(emotion_input[0]['index'],roi)
            emotion_model.invoke()
            #Se hace la predicción
            prediction = emotion_model.get_tensor(emotion_output[0]['index'])
            emotion_label = emotion_classes[prediction.argmax()]
            pos = (x,y)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,emotion_label,pos, font, 3, (0,0,255), 2, cv2.LINE_4)

        ##############################################
        ##############################################

        

        cv2.imshow('Detector of Emotions', frame)
        #Se guardan los datos
        datat.append(round( time.time()-t0, 2))
        dataE.append(emotion_label)
        #data.append([round( time.time()-t0, 2),emotion_label])

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    except (ValueError) as err:
        #print(err)
        pass
        
cap.release()
cv2.destroyAllWindows()

nt=np.array(datat)
nE=np.array(dataE)
#npData = np.matrix(data)
np.savetxt('Data_tflite.csv', [nt,nE], delimiter=';', fmt='%s')
#np.savetxt('Data_tflite.csv', npData, delimiter=';', fmt='%s')
