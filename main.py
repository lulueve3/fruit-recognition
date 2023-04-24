from ultralytics import YOLO
from PIL import Image
import streamlit as st
import cv2
import numpy as np
# Khởi tạo mô hình YOLO
model = YOLO("./best.pt")

st.header("20110467 - Trần Ngọc Hải")
st.header("20110479 - Nguyễn Trung Hiếu")

st.header("['apple', 'banana', 'bell_pepper', 'carrot', 'chilli_pepper', 'corn', 'cucumber', 'eggplant', 'mango', 'orange', 'pineapple', 'potato', 'sweetpotato', 'tomato', 'watermelon']")


uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh và chuyển đổi sang định dạng phù hợp để dự đoán
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (800, 800))
    # Dự đoán các đối tượng có trong ảnh
    results = model(source=image)

    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs

    # In ra số lượng hộp giới hạn được dự đoán

    #Chuyển về đúng hệ màu
    image_result = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh và kết quả dự đoán
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.resize(image, (500,500)), caption="Ảnh đã chọn")

    with col2:
        st.image(cv2.resize(image_result, (500,500)), caption="Kết quả dự đoán")
    
    num_boxes = len(boxes)
    st.write("Số lượng quả: ", num_boxes)
    # Xuất ra tọa độ của từng hộp giới hạn được dự đoán
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        st.write("Tọa độ hộp giới hạn:", (x1, y1), (x2, y2))