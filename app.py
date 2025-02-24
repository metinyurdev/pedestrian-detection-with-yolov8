import cv2
import torch
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import io
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model
model = YOLO("yolov8m.pt").to(device)

# Streamlit app configuration
st.set_page_config(
    page_title="Pedestrian Detection App",
    page_icon="🚶‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Theme Configuration
def get_theme():
    now = datetime.now().hour
    return "day" if 6 <= now < 18 else "night"


def apply_theme(theme):
    if theme == "night":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #40e0d0; /* Turquoise text */
            }
            .stButton>button {
                background-color: #4a90e2;
                color: #ff0000; /* Red button text */
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .stFileUploader>div>div>div>div {
                color: #ff0000; /* Red text for uploader */
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #40e0d0; /* Turquoise headings */
            }
            .stSuccess {
                color: #40e0d0 !important; /* Turquoise success message */
            }
            .footer {
                color: #40e0d0; /* Turquoise footer text */
                text-align: center;
                margin-top: 2rem;
            }
            .footer a {
                color: #40e0d0; /* Turquoise link */
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f0f0f0;
                color: #006400; /* Dark green text */
            }
            .stButton>button {
                background-color: #4a90e2;
                color: #ff0000; /* Red button text */
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .stFileUploader>div>div>div>div {
                color: #ff0000; /* Red text for uploader */
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #006400; /* Dark green headings */
            }
            .stSuccess {
                color: #006400 !important; /* Dark green success message */
            }
            .footer {
                color: #006400; /* Dark green footer text */
                text-align: center;
                margin-top: 2rem;
            }
            .footer a {
                color: #006400; /* Dark green link */
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

theme = get_theme()
apply_theme(theme)

# Streamlit UI
st.title("🚶‍♂️ Pedestrian Detection App")
st.markdown("**Detect pedestrians in your images using YOLOv8!**")

st.markdown("""
    Welcome to the Pedestrian Detection App! This app uses the YOLOv8 model to detect pedestrians in images. 
    Simply upload an image, click the **Detect Pedestrians** button, and the app will highlight any detected pedestrians.
    You can also download the processed image with the detection results.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("Image uploaded successfully!")

    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if st.button("Detect Pedestrians"):
        with st.spinner("Detecting pedestrians..."):
            results = model(image)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"Person {conf:.2f}"

                    if int(box.cls[0]) == 0:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(image, caption="Detected Pedestrians", use_container_width=True, channels="BGR")

        # Download button 
        is_success, buffer = cv2.imencode(".jpg", image)
        if is_success:
            image_bytes = buffer.tobytes()
            st.download_button(
                label="Download Processed Image",
                data=image_bytes,
                file_name="detected_pedestrians.jpg",
                mime="image/jpeg"
            )
        else:
            st.error("Image encoding error!")

# Footer
st.markdown(
    """
    <div class="footer">
        <p>Made with ❤️ by <a href="https://github.com/metinyurdev" target="_blank">Metin Yurduseven</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)