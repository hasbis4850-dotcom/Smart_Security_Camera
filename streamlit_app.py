import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="AI Smart Security Camera", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Smart Security Camera (Mobile & Laptop Edition)")

ALERT_FOLDER = "AI_Security_Alerts"
if not os.path.exists(ALERT_FOLDER):
    os.makedirs(ALERT_FOLDER)

@st.cache_resource
def load_models():
    from ultralytics import YOLO
    import easyocr
    yolo_model = YOLO("yolov8n.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

def process_frame(frame_bgr, model, reader):
    results = model(frame_bgr, conf=0.55)
    detected_any = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            cls = int(box.cls.item())
            name = model.names[cls]
            conf = round(float(box.conf.item()) * 100, 1)

            detected_any = True
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame_bgr, f"ALERT: {name.upper()} {conf}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if name in ["car", "motorcycle"]:
                plate_crop = frame_bgr[int(y1 + (y2 - y1) * 0.4):y2, x1:x2]
                if plate_crop.size > 0:
                    ocr_result = reader.readtext(plate_crop)
                    if ocr_result:
                        plate_text = ocr_result[0][1]
                        cv2.putText(frame_bgr, f"PLATE: {plate_text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        now = datetime.now()
                        current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                        with open(f"{ALERT_FOLDER}/vehicle_logs.txt", "a") as f:
                            f.write(f"[{current_date_time}] Detected: {name.upper()} | Plate: {plate_text}\n")

    return frame_bgr, detected_any

menu = st.sidebar.radio("Navigation Menu", ["📷 Live Security Camera", "📂 Browse Alert Files", "🔒 Privacy Policy"])

if menu == "📷 Live Security Camera":
    st.subheader("Smart AI Surveillance System")
    
    cam_mode = st.radio(
        "Select Camera Mode:",
        ["📱 Mobile & Laptop Browser Camera (Works Everywhere)", "💻 Laptop Continuous Live Video (Localhost Only)"]
    )

    if cam_mode == "📱 Mobile & Laptop Browser Camera (Works Everywhere)":
        st.info("Works on both Mobile Phone and Laptop! Click 'Take Photo' to scan objects or vehicle plates.")
        img_file = st.camera_input("Capture / Scan Frame")
        
        if img_file is not None:
            with st.spinner("AI is analyzing..."):
                model, reader = load_models()
                image = Image.open(img_file)
                frame_rgb = np.array(image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                processed_bgr, detected_any = process_frame(frame_bgr, model, reader)
                result_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                
                st.image(result_rgb, caption="AI Detection Result", use_container_width=True)
                
                if detected_any:
                    photo_name = f"{ALERT_FOLDER}/alert_{int(time.time())}.png"
                    cv2.imwrite(photo_name, processed_bgr)
                    st.success("🚨 Alert Triggered! Photo & Logs saved.")
                else:
                    st.warning("No target object detected with high confidence.")

    else:
        st.info("Use this mode when running directly on your Laptop.")
        run_live = st.checkbox("Start / Stop Continuous Webcam")
        FRAME_WINDOW = st.image([])
        
        if run_live:
            with st.spinner("Starting Laptop Webcam & AI..."):
                model, reader = load_models()
            cap = cv2.VideoCapture(0)
            last_saved_time = 0
            
            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.error("Laptop webcam not found (Use Browser Camera mode on Mobile/Cloud).")
                    break
                
                processed_bgr, detected_any = process_frame(frame, model, reader)
                
                if detected_any:
                    current_time = time.time()
                    if current_time - last_saved_time > 3:
                        photo_name = f"{ALERT_FOLDER}/alert_{int(current_time)}.png"
                        cv2.imwrite(photo_name, processed_bgr)
                        last_saved_time = current_time
                
                frame_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
            cap.release()

elif menu == "📂 Browse Alert Files":
    st.subheader("Saved Alert Photos & Vehicle Logs")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Files"):
            st.rerun()
    with col2:
        if st.button("🗑️ Delete All Photos & Logs"):
            for f in os.listdir(ALERT_FOLDER):
                os.remove(os.path.join(ALERT_FOLDER, f))
            st.success("All photos and logs deleted!")
            time.sleep(1)
            st.rerun()

    st.markdown("### 📋 Vehicle Number Plate Logs")
    log_path = os.path.join(ALERT_FOLDER, "vehicle_logs.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = f.read()
        st.text_area("Logs History", logs if logs else "No logs recorded yet.", height=150)
    else:
        st.info("No vehicle logs recorded yet.")

    st.markdown("### 📸 Captured Alert Images")
    files = [f for f in os.listdir(ALERT_FOLDER) if f.endswith('.png')]
    files.sort(reverse=True)
    if files:
        cols = st.columns(3)
        for idx, file_name in enumerate(files):
            with cols[idx % 3]:
                st.image(os.path.join(ALERT_FOLDER, file_name), caption=file_name, use_container_width=True)
    else:
        st.info("No alert photos captured yet.")

elif menu == "🔒 Privacy Policy":
    st.subheader("🔒 Privacy Policy & Data Security")
    st.write("""
    **1. Multi-Device Secure Architecture:** Supports both local laptop execution and cloud-hosted mobile browser access with user-controlled camera permissions.
    
    **2. Automated AI Processing:** Powered by YOLOv8 for object detection and EasyOCR for automated license plate recognition.
    
    **3. Evidence & Log Management:** Detected alerts and vehicle timestamps are securely recorded inside the `AI_Security_Alerts` directory.
    
    **4. Ethical Surveillance:** Built strictly for personal security monitoring, vehicle tracking, and academic demonstration.
    """)