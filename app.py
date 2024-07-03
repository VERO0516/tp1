import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image, ImageOps
import io
from torchvision import transforms

API_URL = "http://127.0.0.1:8001/api/v1/predict"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=192,
    width=192,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image_data = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(image_data)
    img = img.resize((28, 28))
    img = img.convert('L')

    st.image(img, caption="Input Image", use_column_width=True)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    if st.button("Predict"):
        files = {'file': ('digit.png', buffer, 'image/png')}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"The model predicts this digit is: {prediction}")
        else:
            st.error("Error occurred while making prediction.")
