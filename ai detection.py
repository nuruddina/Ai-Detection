import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("Parasite Egg Detector")



#----------------------------------------------------------------------------------------------
model = tf.keras.models.load_model("sth_2025_tong_cnn.keras", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def boxlocation(img_c, box_size):
    non_zero_points = np.argwhere(img_c > 0)
    if non_zero_points.size == 0:
        return None

    y_min, x_min = np.min(non_zero_points, axis=0)
    y_max, x_max = np.max(non_zero_points, axis=0)

    return [y_min - box_size, y_max + box_size, x_min - box_size, x_max + box_size]

def drawbox(img, label, a, b, c, d, box_size):
    image = cv2.rectangle(img, (c, a), (d, b), (0, 255, 0), 2)
    image = cv2.putText(image, label, (c + box_size, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 1)
    return image

def objectdet(img):
    img = cv2.resize(img, (img.shape[1] // 1, img.shape[0] // 1), interpolation=cv2.INTER_AREA)

    box_size_y, box_size_x = 550, 550
    step_size = 50
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
        label = f"ev: {result:.2f}"
        boxlocat = boxlocation(img_cont, box_size_x // 2)
        if boxlocat:
            img_output = drawbox(img, label, *boxlocat, box_size_x // 2)

    return img_output
#----------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
if uploaded_file is not None:
    try:
        image = np.array(Image.open(uploaded_file))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        st.image(image, caption="Uploaded Image")

        output_img = objectdet(image)
        st.image(output_img, caption="Processed Image")

    except Exception as e:
        st.error(f"Error loading image: {e}")



------------------------------------------------------------------------------------------------------------------------
import cv2
import nampy as np
from keras.models import load_model
from keras.loses import mean_squared_error

path = ""
class_label = ["Artifact","As_fer","As_unfer","Hd","Hn","Hw","Mif","Ov","Tn","Tt"]

class_config = {
    1: (path + "tong.keras",(460,460)),
    2: (path + "Kumming2003CNN.keras",(200,200)),
}

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
    
models = {idx: load_model(cfg[0], custom_objects={'mse' : mse}) for idx, cfg in class_configs.items()}

def drawbox(img, label, a,b,c,d,color):
  image = cv2.rectangle(img, (c,a),(d,b), color,2)
  image = cv2.putText(image, label,(c,a-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
  return image

def compute_iou(box1, box2):
  y1 = max(box1[0], box2[0])
  y2 = min(box1[1], box2[1])
  x1 = max(box1[2], box2[2])
  x2 = min(box1[3], box2[3])
  inter_w = max(0, x2 - x1)
  inter_h = max(0, y2 - y1)
  inter_area = inter_w * inter_h
  box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
  box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
  union_are = box1_area + box2_area - inter_area
  if union_area == 0:
    return 0
  return inter_area / union_area

def nms (detections, iou_threshold):
  nms_dets=［]
  for class_idx in set([d['class_idx'] for d in detections]):
      class_dets = [d for d in detections if d['class_idx'] == class_idx]
      class_dets = sorted(class _dets, key=lambda x: x['score'], reverse=True)
      keep = []
      while class_dets:
        curr = class_dets.pop(0)
        keep. append (curr)
        class _dets = [d for d in class _dets if compute_iou(curr[ 'bbox'], d['bbox']) < iou_threshold]
      nms_dets. extend (keep)
  return nms_dets

def merge_connected_boxes_by_class (detections, merge_iou_threshold):
  merged =［]
  for class_idx in set([d[ 'class_id'] for d in detections]):
    class_dets - [d for d in detections if d[ 'class_idx'] == class_idx]
    used = set
    groups = [0]
    for i, det in enumerate(class_dets):
      if i in used: 
          continue
      group - [det] 
      used. add (i)
      changed = True
      while changed:
        changed = False
        for j, other in enumerate(class_dets):
            if j in used: 
                continue
            if any (compute_iou(d[ 'bbox'], other[ 'bbox']) › merge_iou_threshold for d in group):
              group.append (other)
              used .add(j)
              changed = True
      groups. append (group)
    for group in groups:
      tops = [d['bbox'][0] for d in group]
      bottoms = [d['bbox'][1] for d in group]
      lefts = [d['bbox'][2] for d in group]
      rights = [d['bbox'][3] for d in group]
      merged_box = [min(top), max(bottoms), min(lefts), max(rights)]
      max_score = max(d['score'] for d in group)
      merged.append({"box": merged_box, "class_idx": class_idx, "score": max_score})
  return merged

def objectDet(filepath, threshold, nms_threshold, merge_iou_threshold):
    img = cv2.imread(filepath)
    img_h, img_w = img.shape[:2]
    detection = []
    resize_input_y, resize_input_x = 64, 64
    step_size = 50

    for class_idx, (model, (box_size_y, box_size_x)) in models.items():
        coords = []
        patches = []
        for i in range(0, img_h - box_size_y +1 , step_size):
            for j in range(0, img_w - box_size_x +1, step_size):
                img_patch = img[i:i+box_size_y, j:j+box_size_x]
                img_patch = cv2.resize(img_patch, (resize_input_y, resize_input_x), interpolation=cv2.INTER_AREA)
                patches.append(img_patch)
                coords.append((i, j))
        if not patches:
            continue
        patches = np.array(patches)
        y_out = model.predict(patches, batch_size=64, verbose=0)
        for idx, pred in enmerate(y_out):
            score = pred[1] if len(pred) > 1 else pred[0]
            if score > threshold:
                a, c = coords[idx]
                b, d = a + box_size_y, c + box_size_x
                detection.apped({"bbox": [a, b, c, d], "score": float(score), "class_idx": class_idx})

    nms_detections = nms(detections, iou_threshold=nms_threshold)
    merged_detections = merge_connected_boxes_by_class(nms_detections, merge_iou_threshold=merge_iou_threshold)


    img_output = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255,), (0,255,255), (255,0,225), (255,255,0), (128,128,0), (0,128,128), (128,0,128), (100,100,100,), (200,100,50)]
    for det in merged_detections:
        a, b, c, d = det['bbox']
        class_idx = det['class_idx']
        label = f"{class_label[class_idx]}: {det['score']:.2f}"
        color = colors[class_idx % len(colors)]
        img_output = drawbox(img_output, label, a, b, c, d, color)
    return img_output
     

















