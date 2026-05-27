from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Object Detector", page_icon="🤖", layout="wide")

# Modern and Professional Header Styling
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    text-align: center;
    color: white;
    margin-bottom: 25px;
">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🔎 REAL-TIME OBJECT DETECTOR</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;">Empowering computer vision to analyze and count objects instantly</p>
</div>
""", unsafe_allow_html=True)

# Custom CSS for Buttons, Selectboxes, and Output Cards
st.markdown("""
<style>
    /* Styling for Sidebar Selectbox and Inputs */
    div.stSelectbox>div {
        border: 2px solid #2a5298 !important;
        border-radius: 10px !important;
    }
    
    /* Custom Styling for the Output Cards */
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1e3c72;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .name-card {
        background-color: #eef2f7;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-weight: bold;
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if "object_count" not in st.session_state:
    st.session_state.object_count = 0
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "detected_objects_list" not in st.session_state:
    st.session_state.detected_objects_list = []
if "run_live_feed" not in st.session_state:
    st.session_state.run_live_feed = False

# SIDE BAR - CONTROL PANEL #
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])
file_uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 25)
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20])

# Styling the Run Button
st.sidebar.markdown("<br>", unsafe_allow_html=True)
run = st.sidebar.button("▶️ Run Image Detection")

st.sidebar.markdown("---")
st.sidebar.write("### 🎥 Camera Controls")

# Styled Start and Stop buttons directly under each other
start_cam = st.sidebar.button("📸 Start Live Camera & Count")
stop_cam = st.sidebar.button("🛑 Stop Live Camera Feed")

if start_cam:
    st.session_state.run_live_feed = True
if stop_cam:
    st.session_state.run_live_feed = False

# Layout Setup
col1, col2, col3 = st.columns(3, gap="large")

# COLUMN 1: System Instructions / File Details
with col1:
    st.subheader("📋 Input Details")
    if file_uploaded:
        st.success("File uploaded successfully")
        st.markdown(f"""
        - **File Name:** `{file_uploaded.name}`
        - **File Type:** `{file_uploaded.type}`
        - **File Size:** `{file_uploaded.size / 1024:.2f} KB`
        """)
    else:
        st.info("Awaiting image upload or live feed activation.")
        st.markdown("""
        ### Quick Guide
        1. **Static Mode:** Upload an image from the sidebar and click **Run Image Detection**.
        2. **Live Mode:** Click **Start Live Camera** to trigger automated real-time object tracking.
        3. **Adjustments:** Tune the **Confidence Threshold** if objects are being missed.
        """)

# --- STATIC IMAGE DETECTION LOGIC ---
if file_uploaded and run and not st.session_state.run_live_feed:
    model = YOLO(f"{model_choice}.pt")
    image = Image.open(file_uploaded)
    img_array = np.array(image)
    
    result = model(img_array, conf=confidence/100, max_det=max_det)
    annotated_image = result[0].plot()
    
    names_dict = result[0].names
    found_names = [names_dict[int(box.cls[0])] for box in result[0].boxes]
    
    st.session_state.processed_image = annotated_image
    st.session_state.object_count = len(result[0].boxes)
    st.session_state.detected_objects_list = list(set(found_names))
    st.session_state.source_type = "Static Image Detection"
    st.session_state.detection_done = True

# COLUMN 2: Display Window (Video or Image Rendering)
with col2:
    if st.session_state.run_live_feed:
        st.markdown("### 🎥 Real-Time Stream")
        model = YOLO(f"{model_choice}.pt")
        
        cap = cv2.VideoCapture(0)  # Change index to 1 if using an external webcam
        
        if not cap.isOpened():
            st.error("🚨 Failed to initialize camera hardware interface!")
            st.session_state.run_live_feed = False
        else:
            frame_placeholder = st.empty()
            
            # Re-establishing placeholders in col3 to stream content without layout shifts
            with col3:
                st.markdown("### 📊 Analysis & Output")
                metric_placeholder = st.empty()
                st.markdown("---")
                st.write("#### 🏷️ Detected Object Names:")
                names_placeholder = st.empty()
            
            while st.session_state.run_live_feed:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = model(frame, conf=confidence/100, max_det=max_det)
                annotated_frame = result[0].plot()
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                names_dict = result[0].names
                current_names = [names_dict[int(box.cls[0])] for box in result[0].boxes]
                
                st.session_state.object_count = len(result[0].boxes)
                st.session_state.detected_objects_list = list(set(current_names))
                
                # Render beautifully styled card updates on live feed
                metric_placeholder.markdown(f"""
                <div class="metric-card">
                    <p style="color: #666; margin: 0; font-size: 0.9rem; text-transform: uppercase;">Total Objects Counted</p>
                    <h2 style="color: #1e3c72; margin: 5px 0 0 0; font-size: 2.5rem;">{st.session_state.object_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                with names_placeholder.container():
                    if st.session_state.detected_objects_list:
                        for obj in st.session_state.detected_objects_list:
                            st.markdown(f'<div class="name-card">🔹 {obj.upper()}</div>', unsafe_allow_html=True)
                    else:
                        st.write("*No items found in frame.*")
                        
            cap.release()
            
    elif st.session_state.detection_done and not st.session_state.run_live_feed:
        st.markdown(f"### 🖼️ {st.session_state.source_type}")
        st.image(st.session_state.processed_image, caption="Visual View Analysis", use_container_width=True)
    else:
        st.markdown("### 🖼️ Detection View Window")
        st.info("System idle. Activate an operational mode via the control panel.")

# COLUMN 3: Data Telemetry Dashboard Display (Static Mode Only)
with col3:
    if not st.session_state.run_live_feed:
        st.markdown("### 📊 Analysis & Output")
        if st.session_state.detection_done:
            # Styled Static Card
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #666; margin: 0; font-size: 0.9rem; text-transform: uppercase;">Total Objects Counted</p>
                <h2 style="color: #1e3c72; margin: 5px 0 0 0; font-size: 2.5rem;">{st.session_state.object_count}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.write("#### 🏷️ Detected Object Names:")
            if st.session_state.detected_objects_list:
                for obj in st.session_state.detected_objects_list:
                    st.markdown(f'<div class="name-card">🔹 {obj.upper()}</div>', unsafe_allow_html=True)
            else:
                st.write("*No items found.*")
        else:
            st.warning("No live data streaming. Awaiting analytical pipeline activation.")

# Modern Footer Layout
st.markdown("""
<hr style="border: 1px solid #eef2f7;">
<div style="
    text-align: center; 
    padding: 10px; 
    font-size: 0.85rem; 
    color: #fff; 
    background-color: #1e3c72; 
    border-radius: 10px;
    width: fit-content;
    margin: 20px auto 0 auto;
    padding-left: 20px;
    padding-right: 20px;
">
    Developed with  by <b>BLECA,SmartLabs<sup style="color:#ff4b4b;">TM</sup></b>
</div>
""", unsafe_allow_html=True)