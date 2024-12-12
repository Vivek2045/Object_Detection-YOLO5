import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import easyocr

# Load the YOLO model
MODEL_PATH = "C:\\Users\\Vivek\\OneDrive\\Desktop\\License Plate Detect\\yolov5s.pt"  # Path to your YOLO model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load the YOLO model. Error: {e}")
    model = None

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit app setup
st.title("Object  Detection")
st.write("Upload an image or video to detect objects .")

# File upload
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file and model:
    # Handle image files
    if uploaded_file.type.startswith("image"):
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to numpy array
        image_np = np.array(image)

        # Perform inference
        results = model(image_np)

        # Get the annotated image with bounding boxes
        try:
            annotated_image = np.squeeze(results.render())  # Render annotated image
            # Display the results
            st.image(annotated_image, caption="Detected Objects", use_column_width=True)

            # Extract bounding box coordinates
            detections = results.xyxy[0].cpu().numpy()

            for detection in detections:
                x_min, y_min, x_max, y_max, conf, cls = detection

                # Object Detection
                st.write(f"Detected object class: {int(cls)} with confidence: {conf:.2f}")

                if int(cls) == 0:  # Assuming class 0 is the license plate
                    # Crop the license plate region
                    license_plate = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]

                    # Perform OCR
                    license_plate_text = reader.readtext(license_plate, detail=0)

                    # Display the cropped license plate and text
                    if license_plate_text:
                        st.image(license_plate, caption="Detected object", use_column_width=True)
                        st.write(f"Extracted Number Plate: {' '.join(license_plate_text)}")
                    else:
                        st.write("No text detected .")

        except AttributeError:
            st.error("Error: Unable to process the image with the YOLO model. Please verify the model path or input format.")

    # Handle video files
    elif uploaded_file.type.startswith("video"):
        # Save the uploaded video temporarily
        video_path = Path("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the video frame by frame
        cap = cv2.VideoCapture(str(video_path))
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = model(frame)

            # Render bounding boxes on the frame
            try:
                annotated_frame = np.squeeze(results.render())

                # Extract bounding box coordinates
                detections = results.xyxy[0].cpu().numpy()
                for detection in detections:
                    x_min, y_min, x_max, y_max, conf, cls = detection

                    # Object Detection
                    st.write(f"Detected object class: {int(cls)} with confidence: {conf:.2f}")

                    if int(cls) == 0:  # Assuming class 0 is the license plate
                        # Crop the license plate region
                        license_plate = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                        # Perform OCR
                        license_plate_text = reader.readtext(license_plate, detail=0)

                        # Display the extracted text
                        if license_plate_text:
                            st.write(f"Extracted Number Plate: {' '.join(license_plate_text)}")
                        else:
                            st.write("No text detected on the license plate.")

                # Write the frame to the output video
                out.write(annotated_frame)

                # Display the frame in real-time in the Streamlit app
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            except AttributeError:
                st.error("Error: Unable to process the video frame with the YOLO model. Please verify the model path or input format.")
                break

        cap.release()
        out.release()

        # Display the processed video
        st.video(output_path)

elif not model:
    st.error("Model loading failed. Please check the model path and internet connection.")

# Add instructions
st.write("Ensure the uploaded file contains visible objects and license plates.")