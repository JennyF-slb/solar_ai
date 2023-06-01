import streamlit as st
from streamlit import components
import numpy as np
import base64
from PIL import Image
import tensorflow as tf
import requests
import io
import cv2
from geopy.geocoders import GoogleV3

# Set the page title
st.set_page_config(page_title="ATOS", layout="wide")

# Create a dropdown menu in the sidebar
selected_page = st.sidebar.selectbox("Our Hub", ("Home", "What we do", "Contact us"))


# Create a column layout with two columns
title_column, logo_column = st.columns([5,1])

# Add content to the columns
with title_column:
    st.title(":blue[AuTomatic detection Of Solar roof]")
    st.subheader("Welcome to ATOS! Harness the power of Sunshine !")
    # Add more content to the left column as needed

with logo_column:
    st.image("image/Final_v2.png", width=250)  # Replace "image/Logo.png" with your logo image path

# Function to load and preprocess the image
def load_image(image_bytes):
    img = Image.open(image_bytes).convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

def model_from_json(json_string, custom_objects=None):
    from keras.layers import deserialize_from_json
    return deserialize_from_json(json_string, custom_objects=custom_objects)

# Function to perform image segmentation
def perform_segmentation(image, selected_model):
    # Load the selected Keras model for image segmentation
    if selected_model == "Model: No Patch":
        model_path = "models/model_vgg19_no_patch_checkpoint.h5"
        model = tf.keras.models.load_model(model_path, compile=False)
    elif selected_model == "Model: Patch-Pad-No overlap_2048":
        model_architecture_path = "models/model_vgg19_architecture.json"
        model_weights_path = "models/vgg16unet_patch_pad_no_overlap_weights_2048x2048_full.h5"

        # Load the model architecture from JSON file
        json_file = open(model_architecture_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)

        # Load the model weights
        model.load_weights(model_weights_path)

        # Divide the image by 255
        image = image / 255.0

    elif selected_model == "Model: Patch-Pad-No overlap_2560":
        model_architecture_path = "models/model_vgg19_architecture.json"
        model_weights_path = "models/vgg19unet_patch_pad_no_overlap_weights_2560x2560_full.h5"

        # Load the model architecture from JSON file
        json_file = open(model_architecture_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)

        # Load the model weights
        model.load_weights(model_weights_path)

        # Divide the image by 255
        image = image / 255.0

    else:
        st.error("Invalid model selection")

    # Perform the segmentation prediction
    segmentation = model.predict(image)
    return segmentation

if selected_page == "Home":

    # Replace 'YOUR_API_KEY' with your actual Google Maps API key
    google_maps_api_key = 'AIzaSyCOzsbWrD24xWmtix3u9hlMfRTNf6EM80w'
    # Embed the map in the Streamlit app with full width
    google_maps_component = components.v1.html(
        f"""
        <div style="width: 100%; display: flex; justify-content: center;">
            <iframe src="https://www.google.com/maps/embed/v1/place?key={google_maps_api_key}&q=,&maptype=satellite" width="100%" height="600"></iframe>
        </div>
        """,
        height=600,
        scrolling=False,
    )

    # Add content to the sidebar
    st.sidebar.title("Welcome")

    sidebar_content1 = "- Cutting-edge technology with expertise"
    sidebar_content2 = "- AI algorithms to swiftly analyze aerial imagery and satellite data"
    sidebar_content3 = "- Economic viability and comprehensive consultation services."

    st.sidebar.write(sidebar_content1)
    st.sidebar.write(sidebar_content2)
    st.sidebar.write(sidebar_content3)

elif selected_page == "What we do":
    tab1, tab2 = st.tabs(["With Images", "With Google Maps"])
    button_col1, button_col2 = st.columns(2)
    with tab1:

        # Add content to the About page
        uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])

        if uploaded_image is not None:
            # Display the uploaded image and snapped image side by side
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Add a button to initiate segmentation
            selected_model = st.sidebar.selectbox(
                "Select Model", ("Model: No Patch", "Model: Patch-Pad-No overlap_2048", "Model: Patch-Pad-No overlap_2560")
            )
        else:
            st.text('Please upload your image')

        with button_col1:
        # Code for handling Button 1 click
            if st.sidebar.button("Segment Image"):
                with col2:
                    # Perform image segmentation
                    image = load_image(uploaded_image)
                    segmentation = perform_segmentation(image, selected_model)

                    # Resize the segmentation to match the input image size
                    segmentation = cv2.resize(segmentation[0], (512, 512))

                    # Calculate the percentage of the area covered by the white mask
                    area_covered = np.sum(segmentation == 1)
                    total_pixels = segmentation.shape[0] * segmentation.shape[1]
                    percentage_covered = (area_covered / total_pixels) * 100

                    st.image(segmentation, caption=f"Segmentation Mask (Area covered: {percentage_covered:.2f}%)", use_column_width=True)

    with tab2:
        # Create a geocoder instance
        geolocator = GoogleV3(api_key="AIzaSyCOzsbWrD24xWmtix3u9hlMfRTNf6EM80w")
        # Input for location name or coordinates
        location_input = st.text_input("Enter location name or coordinates (latitude, longitude)")

        if location_input:
            # Geocode the location
            location = geolocator.geocode(location_input)

            # Get the latitude and longitude
            latitude = location.latitude
            longitude = location.longitude

            # Generate the URL for the static satellite image
            map_size = "512x512"  # Adjust the size as needed
            zoom_level = 15  # Adjust the zoom level as needed

            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom_level}&size={map_size}&maptype=satellite&key={geolocator.api_key}"

            # Download the image
            response = requests.get(map_url)
            if response.status_code == 200:
                image_bytes = response.content
            else:
                st.text('Please set your location')

        else:
            # Get the latitude and longitude
            latitude = 30.26
            longitude = -97.74

            # Generate the URL for the static satellite image
            map_size = "512x512"  # Adjust the size as needed
            zoom_level = 15  # Adjust the zoom level as needed

            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom_level}&size={map_size}&maptype=satellite&key={geolocator.api_key}"

            # Download the image
            response = requests.get(map_url)
            if response.status_code == 200:
                image_bytes = response.content

        if image_bytes is not None:
            # Convert image_bytes to TIFF format
            img = Image.open(io.BytesIO(image_bytes))
            tiff_bytes = io.BytesIO()
            img.save(tiff_bytes, format='TIFF')
            tiff_bytes.seek(0)
            # Display the uploaded image and snapped image side by side
            col1, col2 = st.columns(2)

            with col1:
                # Display the snapped image
                st.image(tiff_bytes, caption="Snapped Image", use_column_width=True)
                        # Add a button to initiate segmentation

        with button_col2:
            if st.sidebar.button("Segment Map"):
                with col2:
                    # Convert TIFF image to OpenCV array
                    tiff_array = np.array(bytearray(tiff_bytes.read()), dtype=np.uint8)
                    image_cv2 = cv2.imdecode(tiff_array, cv2.IMREAD_COLOR)

                    # Add batch dimension to the input image
                    image_cv2_batch = np.expand_dims(image_cv2, axis=0)

                    # Perform image segmentation
                    segmentation = perform_segmentation(image_cv2_batch, selected_model)

                    # Resize the segmentation to match the input image size
                    segmentation = cv2.resize(segmentation[0], (512, 512))

                    # Calculate the percentage of the area covered by the white mask
                    area_covered = np.sum(segmentation == 1)
                    total_pixels = segmentation.shape[0] * segmentation.shape[1]
                    percentage_covered = (area_covered / total_pixels) * 100
                    st.image(segmentation, caption=f"Segmentation Mask (Area covered: {percentage_covered:.2f}%)", use_column_width=True)

elif selected_page == "Contact us":
    st.image("image/Team.png", width=1200)  # Replace "image/Logo.png" with your logo image path
