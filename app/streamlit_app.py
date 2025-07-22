import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
model = load_model('models/emotion_cnn.h5')


# Function to preprocess image
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype('float32') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# Streamlit UI
st.set_page_config(page_title="EmotionSense", page_icon="😄", layout="centered")

st.title("🎭 EmotionSense - Real-time Emotion Detection")
st.markdown("Upload an image or use your webcam to detect human emotions using a trained CNN model.")

# Sidebar options
option = st.sidebar.radio("Choose input mode:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = np.array(img.convert('RGB'))
        face = preprocess_image(img)

        prediction = model.predict(face)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]

        st.subheader(f"Detected Emotion: **{emotion}**")
        st.bar_chart(prediction)

elif option == "Use Webcam":
    st.warning("📷 Webcam support in Streamlit is limited. Use the desktop app or run `python src/main.py` instead.")
