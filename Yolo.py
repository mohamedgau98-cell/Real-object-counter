import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase
import av

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Object Detector", page_icon="🤖", layout="wide")

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

st.markdown("""
<style>
    div.stSelectbox>div { border: 2px solid #2a5298 !important; border-radius: 10px !important; }
    .metric-card { background-color: #f8f9fa; border-left: 5px solid #1e3c72; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .name-card { background-color: #eef2f7; border-left: 5px solid #28a745; padding: 15px; border-radius: 8px; margin-bottom: 8px; font-weight: bold; color: #1e3c72; }
</style>
""", unsafe_allow_html=True)

if "object_count" not in st.session_state: st.session_state.object_count = 0
if "detection_done" not in st.session_state: st.session_state.detection_done = False
if "source_type" not in st.session_state: st.session_state.source_type = None
if "detected_objects_list" not in st.session_state: st.session_state.detected_objects_list = []
if "run_live_feed" not in st.session_state: st.session_state.run_live_feed = False

@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(f"{model_name}.pt")

# --- CONTROL PANEL SIDEBAR ---
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])
model = load_yolo_model(model_choice)

file_uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 20)
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
run = st.sidebar.button("▶️ Run Image Detection")

st.sidebar.markdown("---")
st.sidebar.write("### 🎥 Camera Controls")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("📸 Start Live"):
        st.session_state.run_live_feed = True
        st.session_state.detection_done = False  # Zima picha ya nyuma
with col_btn2:
    if st.button("🛑 Stop Live"):
        st.session_state.run_live_feed = False

# --- LAYOUT SETUP ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("📋 Input Details")
    if file_uploaded:
        st.success("File uploaded successfully")
        st.markdown(f"- **File Name:** `{file_uploaded.name}`\n- **File Type:** `{file_uploaded.type}`\n- **File Size:** `{file_uploaded.size / 1024:.2f} KB`")
    else:
        st.info("Awaiting image upload or live feed activation.")

# --- STATIC IMAGE DETECTION LOGIC ---
if file_uploaded and run and not st.session_state.run_live_feed:
    image = Image.open(file_uploaded)
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    result = model(img_array, conf=confidence/100, max_det=max_det, verbose=False)
    annotated_image = result[0].plot()
    
    names_dict = result[0].names
    found_names = [names_dict[int(box.cls[0])] for box in result[0].boxes]
    
    st.session_state.processed_image = annotated_image
    st.session_state.object_count = len(result[0].boxes)
    st.session_state.detected_objects_list = list(set(found_names))
    st.session_state.source_type = "Static Image Detection"
    st.session_state.detection_done = True

# --- LIVE CAMERA PROCESSOR ---
class YOLOProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, conf_thresh, max_detections):
        self.model = yolo_model
        self.conf = conf_thresh
        self.max_det = max_detections
        self.count = 0
        self.current_names = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        result = self.model(img, conf=self.conf, max_det=self.max_det, verbose=False)
        annotated_frame = result[0].plot()
        
        names_dict = result[0].names
        self.current_names = list(set([names_dict[int(box.cls[0])] for box in result[0].boxes]))
        self.count = len(result[0].boxes)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- COLUMN 2: DISPLAY ---
with col2:
    webrtc_ctx = None
    if st.session_state.run_live_feed:
        st.markdown("### 🎥 Real-Time Stream")
        # Tumia seva mbalimbali za bure za Google ili kuongeza uhakika wa kuunganisha kamera cloud
        ice_servers_config = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
        webrtc_ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": ice_servers_config}),
            video_processor_factory=lambda: YOLOProcessor(model, confidence/100, max_det),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    elif st.session_state.detection_done:
        st.markdown(f"### 🖼️ {st.session_state.source_type}")
        st.image(st.session_state.processed_image, caption="Visual View Analysis", use_container_width=True)
    else:
        st.markdown("### 🖼️ Detection View Window")
        st.info("System idle. Activate an operational mode via the control panel.")

# --- COLUMN 3: ANALYSIS (ZOTE MBILI SASA ZINAFANYA KAZI VIZURI) ---
with col3:
    st.markdown("### 📊 Analysis & Output")
    
    if st.session_state.run_live_feed:
        metric_placeholder = st.empty()
        st.markdown("---")
        st.write("#### 🏷️ Detected Object Names:")
        list_placeholder = st.empty()
        
        if webrtc_ctx and webrtc_ctx.video_processor:
            live_count = webrtc_ctx.video_processor.count
            live_items = webrtc_ctx.video_processor.current_names
            
            metric_placeholder.markdown(f"""
            <div class="metric-card">
                <p style="color: #666; margin: 0; font-size: 0.9rem; text-transform: uppercase;">Total Objects Counted</p>
                <h2 style="color: #1e3c72; margin: 5px 0 0 0; font-size: 2.5rem;">{live_count}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if live_items:
                html_str = "".join([f'<div class="name-card">🔹 {obj.upper()}</div>' for obj in live_items])
                list_placeholder.markdown(html_str, unsafe_allow_html=True)
            else:
                list_placeholder.write("*No items found in frame.*")
            st.rerun()
        else:
            metric_placeholder.info("Initializing camera track pipeline...")
            
    elif st.session_state.detection_done:
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
        st.warning("No data streaming. Awaiting analytical pipeline activation.")

# --- FOOTER ---
st.markdown("""
<hr style="border: 1px solid #eef2f7;">
<div style="text-align: center; padding: 10px; font-size: 0.85rem; color: #fff; background-color: #1e3c72; border-radius: 10px; width: fit-content; margin: 20px auto 0 auto; padding-left: 20px; padding-right: 20px;">
    Developed by <b>Gauss de Elim <sup style="color:#ff4b4b;">TM</sup></b>
</div>
""", unsafe_allow_html=True)