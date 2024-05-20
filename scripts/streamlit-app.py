import cv2
import torch
import streamlit as st
from ultralytics import YOLO
# Function to load the PyTorch model (replace 'path/to/your/model.pt' with the actual path)
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to process a frame and display results
def process_frame(frame, model):
    frame = cv2.flip(frame, 1)
    #print(results.type())
    results = model.track(source=frame, show=False)  # Perform object detection with the model
    #print(type(results)[0])
    print(results[0].cpu())
    #cv2.imshow('Frame', results[0].plot())  # Display the processed frame with detections (if applicable)
    frame = results[0].plot()

    # Extract and display relevant information from results (e.g., bounding boxes, class labels, confidence scores)
    
    #if results[0].pandas().xyxy[0].shape[0] > 0:  # Check if any objects were detected
    #    for _, row in results[0].pandas().xyxy[0].iterrows():
    #        xmin, ymin, xmax, ymax, conf, class_id, name = row.values
    #        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    #        label = f"{name} ({conf:.2f})"
    #        cv2.putText(frame, label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    

    return frame

def main():
    st.title("YOLOv8 with 26 keypoints")
    image_placeholder = st.empty()

    # Load the model (replace with the actual path)
    model_path = "D:\\Interns\\Prabhanjan\\extended-keypoints-yolo\\models\\trained-v8s\\train3\\weights\\best.pt"
    model = load_model(model_path)
    #model = YOLO("yolov8l-pose.pt")

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Streamlit loop to capture frames, process, and display
    while (cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_frame(frame.copy(), model)

        # Display the processed frame on Streamlit
        #st.image(processed_frame, channels="BGR")
        image_placeholder.image(processed_frame, channels="BGR")

        #if st.button('exit'):
            #break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
