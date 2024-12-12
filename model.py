import torch
import cv2
import numpy as np



# Download a pre-trained model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5s', 'yolov5s')  # Use 'yolov5s' for a small model


# Load the YOLO model (example for YOLOv5, adjust for YOLOv6)
model = torch.hub.load('ultralytics/yolov5s', 'custom', path='path_to_your_model.pt')  # Update with your model path

def detect_license_plate(image):
    # Convert the image to a format suitable for YOLO model
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(img)

    # Results
    labels = results.names
    boxes = results.xywh[0].cpu().numpy()  # Get bounding boxes
    return boxes, labels
