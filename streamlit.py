import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os


model = tf.keras.models.load_model('my_model.h5')

mclass = ['Corn: Common Rust',
  'Corn: Gray Leaf Spot',
  'Corn: Healthy',
  'Corn: Leaf Blight',
  'Potato: Early Blight',
  'Potato: Healthy',
  'Potato: Late Blight',
  'Rice: Brown Spot',
  'Rice: Healthy',
  'Rice: Hispa',
  'Rice: Leaf Blast',
  'Wheat: Brown Rust',
  'Wheat: Healthy',
  'Wheat: Yellow Rust']
# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((227, 227))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
  
# Streamlit
st.title('CNN Model Deployment')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    pred = np.argmax(prediction)
    P = mclass[pred]
    st.success(f'Disease Prediction : {P}')
