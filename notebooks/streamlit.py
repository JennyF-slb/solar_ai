import os
import streamlit as st
from streamlit import components
from data import Inria
# from utils import *

def main():

    # Add title and description
    st.title('Interactive Map Snapping')

   # Replace 'YOUR_API_KEY' with your actual Google Maps API key
    google_maps_api_key = 'AIzaSyCOzsbWrD24xWmtix3u9hlMfRTNf6EM80w'
    google_maps_component = components.v1.html(
        # f'<iframe src="https://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}"></iframe>',
        f'<iframe src="https://www.google.com/maps/embed/v1/place?key={google_maps_api_key}&q=Space+Needle,Seattle+WA&maptype=satellite" width="100%" height="600"></iframe>',
        width=700,
        height=600,
        scrolling=True,
    )

    st.markdown('<h1>Google Maps Demo</h1>', unsafe_allow_html=True)
    st.markdown(google_maps_component, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
