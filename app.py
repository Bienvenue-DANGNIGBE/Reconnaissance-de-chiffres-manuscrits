#######================================================================########
#######                                                                ########
#######                PROJET MACHINE LEARNING                         ########
#######          Reconnaissance de chiffres manuscrits                 ########
#######           (REALISE PAR : BIENVENUE DANGNIGBE)                  ########
#######                                                                ########
#######================================================================########

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# =================== Entraînement du modèle CNN ===================

@st.cache_resource
def train_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=8, batch_size=128, verbose=0)
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    return model, accuracy


model, accuracy = train_model()


# =================== Prétraitement de l'image ===================

def preprocess_image(image):
    img = np.array(image.convert("L"))

    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (20,20))
    canvas = np.zeros((28,28), dtype=np.uint8)
    canvas[4:24, 4:24] = img

    img = canvas / 255.0
    img = img.reshape(1,28,28,1)

    return img


# =================== Interface Streamlit ===================

st.title("Reconnaissance de chiffres manuscrits")
st.markdown("**CNN + prétraitement avancé (Base MNIST)**")
st.success(f"Précision MNIST : {accuracy*100:.2f}%")

uploaded_file = st.file_uploader(
    "Chargez une image manuscrite (fond blanc recommandé)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed = preprocess_image(image)

    prediction = model.predict(processed)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(image, caption="Image originale", width=200)
    st.success(f"Chiffre détecté : **{digit}**")
    st.info(f"Confiance : **{confidence*100:.2f}%**")
