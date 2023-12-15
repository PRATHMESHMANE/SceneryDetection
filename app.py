import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from utils import get_models, get_image_file,  mediapipe_detection, opencv_detection
from utils import mp_face_detection, mp_drawing, emotion_dict, rescale, enhance_image, enhance_image_with_adaptive_histogram_equalization, enhance_image_with_histogram_equalization, enhance_image_with_unsharp_masking, enhance_image_with_bilateral_filter
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 330px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 330px
        margin-left: -350px
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div:first-child h1{
        padding: 0rem 0rem 0rem 0rem;
        text-align: center;
        font-size: 2rem;
    }
    .css-1544g2n.e1fqkh3o4 {
        padding-top: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Basic App Scaffolding
st.title('Scene Detection')
st.divider()

with st.sidebar:
    st.title('Scenery')
    st.divider()
    # Define available pages in selection box
    app_mode = option_menu("", [ "Image"],
                           icons=["images"], menu_icon="list", default_index=0,
                           styles={
                               "icon": {"font-size": "1rem"},
                               "nav-link": {"font-family": "roboto", "font-size": "1rem", "text-align": "left"},
                               "nav-link-selected": {"background-color": "tomato"},
                           }
                           )


# Image Page
if app_mode == 'Image':

    # Sidebar

    model = get_models()
    detection_type = st.sidebar.radio('',['Scene Detection', 'Indiviual'])
    st.sidebar.divider()

    detection_confidence = 0.5


    image = get_image_file()



    # Enhance the inputed image
    # image = enhance_image(image=image)
    # image = enhance_image_with_adaptive_histogram_equalization(image)
   
    st.sidebar.write('Image Enhancement Options')
    enhancement_option = st.sidebar.selectbox('Select Enhancement Technique', ['None', 'Histogram Equalization', 'Adaptive Histogram Equalization', 'Unsharp Masking', 'Bilateral Filter'])
    if enhancement_option == 'Histogram Equalization':
        enhanced_image = enhance_image_with_histogram_equalization(image)
    elif enhancement_option == 'Adaptive Histogram Equalization':
        enhanced_image = enhance_image_with_adaptive_histogram_equalization(image)
    elif enhancement_option == 'Unsharp Masking':
        enhanced_image = enhance_image_with_unsharp_masking(image)
    elif enhancement_option == 'Bilateral Filter':
        enhanced_image = enhance_image_with_bilateral_filter(image)
    else:
        enhanced_image = image


    if detection_type == 'Scene Detection':
        mediapipe_detection(detection_confidence, enhanced_image, model, 'With full image')
    else:
        opencv_detection(enhanced_image, model, 'With full image')
    
