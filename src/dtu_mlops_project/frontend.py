import os

import mimetypes
import pandas as pd
from PIL import Image
import requests
import streamlit as st
from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/dtu-mlops-447711/locations/europe-west4"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


def about_model(backend):
    """Send a request to the backend to get information about the model."""
    about_url = f"{backend}/about/"
    response = requests.get(about_url)
    if response.status_code == 200:
        return response.json()
    return None


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/api/predict/"
    files = {'image_file': open(image.name, 'rb')}
    response = requests.post(predict_url, files=files, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification with MobileNetV4")

    # Introduction
    st.write("""
        This application uses a fine-tuned MobileNetV4 model, sourced from PyTorch Image Models (TIMM) utilizing the ImageNet dataset.
        You can upload an image, and the model will predict the type of vehicle in the image.
    """)

    # add button to get information about the model
    st.header("Model Used")
    model_info = about_model(backend)
    if model_info is not None:
        st.write(model_info["model_name"])
    else:
        st.write("Failed to get model information")

    # Instructions
    st.header("Instructions")
    st.write("""
        1. Click on the "Browse files" button to upload an image.
        2. The image should be in JPEG, PNG, SVG, or SVG+XML format.
        3. Once uploaded, the model will classify the image and display the results.
    """)

    # File Uploader
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "png", "svg", "svg+xml"])

    if uploaded_file is not None:
        # Get the MIME type of the uploaded file
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)

        # Check if the MIME type is in the accepted types
        if mime_type not in ["image/jpeg", "image/png", "image/svg+xml"]:
            st.write("Invalid file type! Please upload an image of type JPEG, PNG, or SVG.")

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        result = classify_image(uploaded_file, backend=backend)

        if result is not None:
            probabilities = result["probabilities"]

            # Results
            st.header("Results")
            st.write("Prediction Probabilities:")

            # Bar Chart
            data = {"Class": list(probabilities.keys()), "Probability": list(probabilities.values())}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to classify the image")

    # Resources
    st.header("Resources")
    st.write("""
        - [PyTorch Image Models (TIMM) Documentation](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#getting-started-documentation)
        - [MobileNet-V4](https://huggingface.co/blog/rwightman/mobilenetv4)
        - [Dataset imagenet-1k-wds (subset)](https://huggingface.co/datasets/timm/imagenet-1k-wds)
    """)


if __name__ == "__main__":
    main()
