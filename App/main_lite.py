import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
#from skimage.transform import resize
import time

def crop_center(img, x, y, w, h):    
    return img[y:y+h,x:x+w]

def preprocess_img(raw):
    #img = cv2.resize(raw,(200,200,3))
    #img = resize(raw,(200,200,3))
    #print("raw")
    #print(raw)
    img = np.resize(raw,(200,200,3)) 
    img=img/255
    #print("img")
    #print(img)
    
    #print("ellos")
    #img2 = resize(raw,(200,200,3))
    #print(img2)
    
    #print("Son iguales")
    #print(img==img2)
    
    #img=img/258.6
    #print("img1")
    #print(img1)
    #print("nosotros")
    #print(img)
    #print(img1==img)
    #img = cv2.resize(raw,(200,200,3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    #print(img)
    return img

def brain(raw, x, y, w, h):
    ano = ''
    img = crop_center(raw, x, y , w , h)
    img = preprocess_img(img)
    f.set_tensor(i['index'], img.astype(np.float32))
    f.invoke()
    res = f.get_tensor(o['index'])
    classes = np.argmax(res,axis=1)
    print(classes)
    if classes == 0:
        ano = 'anger'
    elif classes == 1:
        ano = 'disgust'
    elif classes == 2:
        ano = 'fear'
    elif classes == 3:
        ano = "happy"
    elif classes == 4:
        ano = "neutral"
    elif classes == 5:
        ano = 'sadness'
    else :
        ano = 'surprised'
    return ano
    

print('Loading ..')

f = tflite.Interpreter("model_optimized.tflite")
f.allocate_tensors()
i = f.get_input_details()[0]
o = f.get_output_details()[0]

print('Load Success')

cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)


cap = cv2.VideoCapture(0)
ai = 'anger'
img = np.zeros((200, 200, 3))
ct = 0
#data=np.asarray([["Time(s)","Emotion"]])
t0 = time.time()
datat = ['Time (s)'] #Se guarda el dato del tiempo
dataE = ['Emotion'] #Se guarda el dato de la emoción


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ct+=1
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    
    ano = ''    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, ai, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        if ct > 3:
            ai = brain(gray, x, y, w, h)
            ct = 0
            #np.insert(data,round( time.time()-t0, 2),ai)
            #data.insert([round( time.time()-t0, 2),ai])
            datat.append(round( time.time()-t0, 2))
            dataE.append(ai)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
#np.savetxt('Data_tflite.csv', data, delimiter=';', fmt='%s')
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

nt=np.array(datat)
nE=np.array(dataE)
#npData = np.matrix(data)
np.savetxt('Data_tflite.csv', [nt,nE], delimiter=';', fmt='%s')








