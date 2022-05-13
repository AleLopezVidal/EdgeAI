from keras.models import load_model
import keras
import cv2
import numpy as np
import tensorflow as tf


face_classifier=cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
#Se obtiene modelo de:
#https://github.com/karansjc1/emotion-detection/blob/master/without%20flask/EmotionDetectionModel.h5
emotion_model = load_model('../EmotionDetectionModel.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
tflite_emotion = converter.convert()
open('emotion.tflite','wb').write(tflite_emotion)
