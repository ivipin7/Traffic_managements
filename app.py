import os
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Set environment variable to avoid duplicate library issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Load and preprocess the image from an .npy file
def load_image(image_path):
    # Load the numpy array from the file
    image = np.load(image_path)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)  # Convert to uint8 if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    else:
        raise ValueError("Expected a 3-channel image (RGB).")
    return image_bgr

# Step 2: Detect vehicles using YOLOv8 model
def detect_vehicles(image_bgr):
    model = YOLO('yolov8n.pt')  # YOLOv8 nano model
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)  # Get detection results
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data
    return detections

# Step 3: Draw bounding boxes, count vehicles, and calculate accuracy
def draw_boxes_and_calculate_accuracy(image_bgr, detections):
    vehicle_classes = [2, 3, 5, 7]  # Define vehicle class IDs
    vehicle_count = 0
    total_confidence = 0.0
    for det in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = det
        class_id = int(class_id)
        if class_id in vehicle_classes:
            vehicle_count += 1
            total_confidence += confidence
            label = f"Vehicle ({confidence:.2f})"
            # Draw bounding boxes and labels
            cv2.rectangle(image_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    average_confidence = total_confidence / vehicle_count if vehicle_count > 0 else 0.0
    return image_bgr, vehicle_count, average_confidence

# Step 4: Calculate signal time
def calculate_signal_time(vehicle_count, base_time=10, time_per_vehicle=2, max_time=60):
    signal_time = base_time + (vehicle_count * time_per_vehicle)
    return min(signal_time, max_time)

# Streamlit Web App
def main():
    st.title("AI Powered Real Time Road Traffic Optimization")
    st.sidebar.title("Options")

    # Upload Image
    uploaded_file = st.sidebar.file_uploader("Upload a .npy file", type=["npy"])

    if uploaded_file:
        with st.spinner("Loading image..."):
            # Load the image
            try:
                # Temporary location for uploaded file
                with open("temp_image.npy", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_bgr = load_image("temp_image.npy")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return

        # Detect vehicles
        with st.spinner("Detecting vehicles..."):
            detections = detect_vehicles(image_bgr)

        # Draw bounding boxes and calculate accuracy
        image_with_boxes, vehicle_count, average_confidence = draw_boxes_and_calculate_accuracy(image_bgr, detections)

        # Calculate signal time
        signal_time = calculate_signal_time(vehicle_count)

        # Display Results
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Vehicles")
        st.markdown(f"### Vehicle Count: {vehicle_count}")
        st.markdown(f"### Average Accuracy: {average_confidence:.2f}")
        st.markdown(f"### Recommended Signal Time: {signal_time} seconds")

        # Save Processed Image Option
        if st.button("Save Processed Image"):
            output_path = "processed_image.jpg"
            cv2.imwrite(output_path, image_with_boxes)
            st.success(f"Image saved as {output_path}")

if __name__ == "__main__":
    main()
