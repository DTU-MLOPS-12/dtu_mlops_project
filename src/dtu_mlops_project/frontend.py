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


def classify_image_model(image, backend, dummy_model):
    """Send the image to the backend for classification."""
    if dummy_model:
        predict_url = f"{backend}/api/predict/dummy/"
    else:
        predict_url = f"{backend}/api/predict/"
    files = {"image_file": ("uploaded_image", image, image.type)}
    # files = {"image_file": open(image.name, "rb")}
    response = requests.post(predict_url, files=files, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if not backend:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification with MobileNetV4")

    # Introduction
    st.write("""
        This application uses a fine-tuned MobileNetV4 model, sourced from PyTorch Image Models (TIMM) utilizing the ImageNet dataset.
        You can upload an image, and the model will predict the type of vehicle in the image.
    """)

    # add button to get information about the model
    st.header("Trained Model Used")
    model_info = about_model(backend)
    if model_info is not None:
        st.write(model_info["model_name"])
    else:
        st.write("Failed to get model information")

    # Instructions
    st.header("Instructions")
    st.write("""
        1. Choose the model to use (Trained Model or Dummy Model).
        2. Upload an image of a vehicle (car, truck, etc.). Only JPEG, PNG, and SVG images are supported.
        3. Click the 'Classify' button to get the prediction results.
    """)

    # Choose dummy model or Trained model
    st.header("Choose Model")
    model_choice = st.selectbox(
        "Select the model to use", ["Trained Model", "Dummy Model"], index=None, placeholder="Select model..."
    )
    if model_choice == "Dummy Model":
        dummy_model = True
    else:
        dummy_model = False

    # Show the file uploader only if a model has been chosen
    if model_choice:
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

            result = classify_image_model(uploaded_file, backend=backend, dummy_model=dummy_model)

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
