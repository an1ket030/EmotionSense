
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def load_emotion_model(model_path='models/emotion_cnn.h5'):
    return load_model(model_path)

def predict_emotion(model, face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    preds = model.predict(face_img)[0]
    return preds
