import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )
    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Leather Product Images."
    )

# Image loading helper
def load_uploaded_image(file):
    img = Image.open(file).convert("RGB")
    return img

# Input method selection
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

# Image input logic
image_input = None
if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_input = load_uploaded_image(uploaded_file)
        st.image(image_input, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        image_input = load_uploaded_image(camera_image_file)
        st.image(image_input, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anomaly Detection function
def Anomaly_Detection(image_pil):
    model_path = "./weights/leather_model.pt"  # Make sure this file exists
    class_names = ['Anomaly', 'Good']
    threshold = 0.5

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image_pil).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    # Prediction logic
    predicted_class_index = int(prob > threshold)
    predicted_class = class_names[predicted_class_index]

    # Message
    if predicted_class == "Good":
        return "✅ Congratulations! Your product has been classified as a 'Good' item with no anomalies detected."
    else:
        return "⚠️ Anomaly detected in your product. Please review the inspection image."

# Submit button logic
submit = st.button(label="Submit a Leather Product Image")
if submit:
    st.subheader("Output")
    if image_input is not None:
        with st.spinner("Analyzing image, please wait..."):
            prediction = Anomaly_Detection(image_input)
            st.success(prediction)
    else:
        st.error("Please provide an image before submitting.")
