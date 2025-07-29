import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

def ai_detector_page():
    st.title("AI Detector")
    st.subheader("Upload & View Image")
    st.write("Upload an image and view it below.")

    #----------------------------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "sth_2025_tong_cnn.keras")
    #----------------------------------------------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )

    def boxlocation(img_c, box_size):
        non_zero_points = np.argwhere(img_c > 0)
        if non_zero_points.size == 0:
            return None
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)
        return [y_min - box_size, y_max + box_size, x_min - box_size, x_max + box_size]

    def drawbox(img, label, a, b, c, d, box_size):
        image = cv2.rectangle(img.copy(), (c, a), (d, b), (0, 255, 0), 2)
        image = cv2.putText(image, label, (c + box_size, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)
        return image

    def objectdet(img):
        box_size_y, box_size_x = 460, 460
        step_size = 150
        img_output = np.array(img)
        img_cont = np.zeros((img_output.shape[0], img_output.shape[1]), dtype=np.uint8)
        result = 0

        for i in range(0, img_output.shape[0] - box_size_y, step_size):
            for j in range(0, img_output.shape[1] - box_size_x, step_size):
                img_patch = img_output[i:i + box_size_y, j:j + box_size_x]
                img_patch = cv2.resize(img_patch, (64, 64), interpolation=cv2.INTER_AREA)
                img_patch = np.expand_dims(img_patch, axis=0)

                y_outp = model.predict(img_patch, verbose=0)

                if result < y_outp[0][1] and y_outp[0][1] > 0.95:
                    result = y_outp[0][1]
                    img_cont[i + (box_size_y // 2), j + (box_size_x // 2)] = int(y_outp[0][1] * 255)

        if result != 0:
            label = f"Ov: {result:.2f}"
            boxlocat = boxlocation(img_cont, box_size_x // 2)
            if boxlocat:
                img_output = drawbox(img, label, *boxlocat, box_size_x // 2)

        return img_output

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            st.image(image, caption="Uploaded Image", use_container_width=True )

            output_img = objectdet(image)
            st.image(output_img, caption="Processed Image", use_container_width=True )

        except Exception as e:
            st.error(f"Error loading image: {e}")

#================= เรียกใช้งานฟังก์ชัน ===================
if __name__ == "__main__":
    ai_detector_page()
