import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# TensorFlow Model Prediction
@st.cache_resource  # Cache the model to avoid reloading it every time
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

# Prediction function
def model_prediction(test_image):
    # Load the model once at the start
    model = load_model()

    # Convert the uploaded file into an image
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to match the model input
    input_arr = np.array(image)  # Convert to NumPy array
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch of size 1

    predictions = model.predict(input_arr)
    
    # Sort predictions to get top 5
    top_per_5 = np.sort(predictions).flatten()[::-1]
    top_5 = np.argsort(predictions).flatten()[::-1]

    # Disease labels
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                   'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                   'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

    # Prepare result (Top 5 predictions)
    disease = top_5[:5]
    percentage = top_per_5[:5] * 100
    title_disease = "\n".join([f"{class_names[disease[i]]}: {percentage[i]:.2f}%\n" for i in range(5)])

    return title_disease  # Return the formatted string

# Image segmentation
def segment_diseased_area(image):
    """
    Segments the diseased area in a given image.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array (RGB format).
    
    Returns:
        numpy.ndarray: The combined image showing the original, mask, and contour results.
    """
    # Ensure the image is in RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] != 3 else image

    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range for 'diseased' colors (e.g., brown, yellow)
    lower_bound = np.array([10, 50, 50])  # Adjust values based on the disease color
    upper_bound = np.array([35, 255, 255])

    # Create a mask for the diseased region
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Find contours of the masked area
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

    # Convert mask to a 3-channel image to concatenate
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Stack images horizontally
    combined_image = np.hstack((image, mask_3_channel, contour_image))

    return combined_image  # Return combined image for display

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "Picture1.JPG"  # Update this path
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset contains images of healthy and diseased crop leaves categorized into 38 different classes.
    """)

# Disease Recognition
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        # Display uploaded image
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        # Predict button
        if st.button("Predict"):
            #st.snow()  # Show a snowfall effect while processing

            # Model prediction
            prediction_result = model_prediction(test_image)
            st.write("Model Prediction Result: \n", prediction_result)

            # Image segmentation
            image = Image.open(test_image)
            image = np.array(image)
            combined_image = segment_diseased_area(image)
            st.image(combined_image, caption="Diseased Areas", use_container_width=True)
