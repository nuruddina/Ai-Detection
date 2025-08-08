import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

MODEL_PATH = "C:/Users/Mufha/Downloads/best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLO Object Detection")
st.markdown("Upload an image to run object detection using YOLO. Class 0 will be hidden.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    cv2.imwrite(temp_path, image)

    results = model(temp_path, conf=conf_threshold)

    boxes = results[0].boxes.data.cpu().numpy()
    filtered_boxes = []
    for det in boxes:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:
            filtered_boxes.append(det)

    annotated_img = image.copy()
    for det in filtered_boxes:
        x1, y1, x2, y2, conf, cls = det
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(annotated_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.subheader("✅ Detection Result (Class 0 Hidden)")
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("📄 Detection Details")
    for det in filtered_boxes:
        x1, y1, x2, y2, conf, cls = det
        st.write(f"Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, "
                 f"Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
