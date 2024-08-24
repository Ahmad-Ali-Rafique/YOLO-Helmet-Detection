import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Sidebar content
st.sidebar.image('Ahmad Ali.png', use_column_width=True)
st.sidebar.header("**Ahmad Ali Rafique**")
st.sidebar.write("AI & Machine Learning Expert")

st.sidebar.header("About Model")
st.sidebar.info('''This Model is designed for real-time helmet detection using the YOLOv3 (You Only Look Once) model  
1️⃣ Click on Upload button   
2️⃣ Upload images to detect helmets  
3️⃣ See the results''')

st.sidebar.header("Contact Information")
st.sidebar.write("Feel free to reach out through the following")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)")
st.sidebar.write("[GitHub](https://github.com/Ahmad-Ali-Rafique/)")
st.sidebar.write("[Email](mailto:arsbussiness786@gmail.com)")
st.sidebar.write("Developed by Ahmad Ali Rafique", unsafe_allow_html=True)

# Check if YOLO files are present
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
names_path = "coco.names"

if not os.path.exists(weights_path) or not os.path.exists(cfg_path) or not os.path.exists(names_path):
    st.error("YOLO files are missing. Please download the following files:")
    st.markdown(f"""
    - [Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
    - [Download YOLOv3 Config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - [Download COCO Names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
    """)
else:
    # Load YOLO
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()

    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    def detect_objects(image):
        # Convert the image to a format suitable for OpenCV
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                confidence_label = f"{confidences[i]:.2f}"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(img, confidence_label, (x + w - 50, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    st.title("YOLO Helmet Detection")
    st.write("Upload an image to detect helmets.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Detecting...")
        result_image = detect_objects(image)
        st.image(result_image, caption="Processed Image.", use_column_width=True)
