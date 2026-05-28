from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase
import av

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

# Hint to user for far objects: lower confidence helps detect distant small frames
confidence = st.sidebar.slider("Confidence Threshold (Lower this for distant objects)", 0, 100, 20)
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
    
    # Static Image Zoom Pipeline for far object scaling
    h, w = img_array.shape[:2]
    img_resized = cv2.resize(img_array, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC) if w < 1000 else img_array
    
    result = model(img_resized, conf=confidence/100, max_det=max_det)
    annotated_image = result[0].plot()
    
    # Scale back down for standard display view consistency
    if w < 1000:
        annotated_image = cv2.resize(annotated_image, (w, h), interpolation=cv2.INTER_AREA)
    
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
        
        @st.cache_resource
        def load_yolo_model(model_name):
            return YOLO(f"{model_name}.pt")
        
        model = load_yolo_model(model_choice)

        # Class based video processor with Auto-Scaling for distant objects
        class YOLOProcessor(VideoProcessorBase):
            def __init__(self):
                self.count = 0
                self.current_names = []

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                
                # Dynamic Frame Scaling Pipeline to enhance far/small objects feature extraction
                h, w = img.shape[:2]
                scaled_img = cv2.resize(img, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
                
                result = model(scaled_img, conf=confidence/100, max_det=max_det)
                annotated_frame = result[0].plot()
                
                # Downscale back to original aspect ratio for presentation window
                final_frame = cv2.resize(annotated_frame, (w, h), interpolation=cv2.INTER_AREA)
                
                names_dict = result[0].names
                self.current_names = list(set([names_dict[int(box.cls[0])] for box in result[0].boxes]))
                self.count = len(result[0].boxes)
                
                return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

        # Robust multi-server STUN/TURN configurations to bypass carrier grade firewalls/NATs
        ice_servers_config = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]}
        ]

        # Start WebRTC Streamer with powerful bypass configurations
        webrtc_ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": ice_servers_config}),
            video_processor_factory=YOLOProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
    elif st.session_state.detection_done and not st.session_state.run_live_feed:
        st.markdown(f"### 🖼️ {st.session_state.source_type}")
        st.image(st.session_state.processed_image, caption="Visual View Analysis", use_container_width=True)
    else:
        st.markdown("### 🖼️ Detection View Window")
        st.info("System idle. Activate an operational mode via the control panel.")

# COLUMN 3: Data Telemetry Dashboard Display (Static & Live support)
with col3:
    st.markdown("### 📊 Analysis & Output")
    
    if st.session_state.run_live_feed and webrtc_ctx.video_processor:
        live_count = webrtc_ctx.video_processor.count
        live_items = webrtc_ctx.video_processor.current_names
        
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #666; margin: 0; font-size: 0.9rem; text-transform: uppercase;">Total Objects Counted</p>
            <h2 style="color: #1e3c72; margin: 5px 0 0 0; font-size: 2.5rem;">{live_count}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.write("#### 🏷️ Detected Object Names:")
        if live_items:
            for obj in live_items:
                st.markdown(f'<div class="name-card">🔹 {obj.upper()}</div>', unsafe_allow_html=True)
        else:
            st.write("*No items found in frame.*")
            
    elif not st.session_state.run_live_feed and st.session_state.detection_done:
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
    Developed by <b>BLECA,SmartLabs<sup style="color:#ff4b4b;">TM</sup></b>
</div>
""", unsafe_allow_html=True)