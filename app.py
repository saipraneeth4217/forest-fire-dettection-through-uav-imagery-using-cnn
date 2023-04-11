import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained CNN model
model = tf.keras.models.load_model(r"C:\Users\sandeep\OneDrive\Desktop\new_model\forest_fire.h5")

# Set up the Streamlit app
st.set_page_config(page_title='Forest Fire Detection',
                   page_icon='🔥',
                   layout='wide')

st.title('Forest Fire Detection through UAV Imagery')

# Allow the user to upload an image
file = st.file_uploader('Upload an image of the forest:', type=['jpg', 'jpeg', 'png'])

if file is not None:
    # Load the image and preprocess it
    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Use the model to make a prediction
    prediction = model.predict(img_array)
    prob_burnt = prediction[0][0]
    threshold = 0.5  # Set the threshold based on your dataset

    # Determine the result based on the threshold
    if prob_burnt > threshold:
        result = 'Burnt'
    else:
        result = 'Not burnt'

    # Display the original image and the prediction result
    st.image(img, caption='Original Image', use_column_width=True)
    st.write('The predicted result is:', result)
    st.write('Probability of being burnt:', prob_burnt)