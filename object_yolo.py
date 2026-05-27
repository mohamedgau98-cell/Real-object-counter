from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="OBJECT DETECTOR", page_icon="🤖", layout="wide")
st.markdown("""
<div style="
border:2px solid blue;
border-radius:20px;
width:100%;
background-color:violet;
text-align:center;
">
<center>
<h2>🔎 OBJECT DETECTOR</h2>
<p>Upload image of object and let AI find the object for you</p>
</center>
</div>
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

# SIDE BAR #
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])
file_uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 25)
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20])

# Run Detection Button (For Uploaded Images)
run = st.sidebar.button("Run Detection")

st.sidebar.markdown("---")
st.sidebar.write("### 🎥 Live Camera Controls")

# Trigger Live continuous tracking instantly
if st.sidebar.button("📸 Take Picture & Count Automatically"):
    st.session_state.run_live_feed = True

# STOP button placed right underneath camera activation
if st.sidebar.button("🛑 Stop Live Camera Feed"):
    st.session_state.run_live_feed = False

st.markdown("""
<style>
    div.stButton>button:first-child{
        background-color:blue;
        color:white;
        width: 100%;
    }
    div.stSelectbox>div{
        border:2px solid red;
        border-radius:20px;
        background-color:blue;
        size:20px;
        color:white;
    }
</style>
""", unsafe_allow_html=True)

# Main Layout #
col1, col2, col3 = st.columns(3, gap="large")

# Column 1: Instructions / File Details
with col1:
    st.subheader("Input Preview")
    if file_uploaded:
        st.info("File uploaded successfully")
        st.write(f"File Name: {file_uploaded.name}")
        st.write(f"File Type: {file_uploaded.type}")
        st.write(f"File size: {file_uploaded.size}")
    else:
        st.info("Upload The Image from the side bar")
        st.markdown("""
### Instructions
#### 1. Upload an image or toggle Live Count
#### 2. Choose a model
#### 3. Set Confidence level
#### 4. Monitor results on the columns
""")

# footer
st.markdown("""
<style>
    .footer{
        position:relative;
        left:0;
        bottom:0;
        width:fit-content;
        border-radius:20px;
        background-color:blue;
        color:white;
        text-align:center;
        padding:10px;
        font-size:12px;
    }
</style>
<div class="footer">
    Developed by BLECA<sup style="color:red">TM</sup>
</div>
""", unsafe_allow_html=True)

# Shared logic for Static Upload
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

# Column 2: Camera or Static Image View
with col2:
    if st.session_state.run_live_feed:
        st.markdown("### 🎥 Live Video Streaming")
        model = YOLO(f"{model_choice}.pt")
        
        # Uses video capture profile index 0 (Change to 1 if using external camera)
        cap = cv2.VideoCapture(0)  
        
        if not cap.isOpened():
            st.error("🚨 Failed to open Camera hardware interface!")
            st.session_state.run_live_feed = False
        else:
            frame_placeholder = st.empty()
            
            # Placeholders inside col3 to prevent duplication during live looping
            with col3:
                st.markdown("### 📊 Analysis & Object Count")
                metric_placeholder = st.empty()
                st.write("---")
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
                
                # Update output area cleanly without duplicating UI elements
                metric_placeholder.metric(label="Total Objects Counted", value=st.session_state.object_count)
                
                with names_placeholder.container():
                    if st.session_state.detected_objects_list:
                        for obj in st.session_state.detected_objects_list:
                            st.markdown(f"- **{obj.upper()}**")
                    else:
                        st.write("*No items found.*")
                        
            cap.release()
            
    elif st.session_state.detection_done and not st.session_state.run_live_feed:
        st.markdown(f"### 🖼️ {st.session_state.source_type}")
        st.image(st.session_state.processed_image, caption="Processed Visual Output", use_container_width=True)
    else:
        st.markdown("### 🖼️ Image Detection Window")
        st.info("Awaiting detection input from the Control Panel.")

# Column 3: Telemetry Analysis Output (Static Mode Display Only)
with col3:
    if not st.session_state.run_live_feed:
        st.markdown("### 📊 Analysis & Object Count")
        if st.session_state.detection_done:
            st.metric(label="Total Objects Counted", value=st.session_state.object_count)
            st.write("---")
            st.write("#### 🏷️ Detected Object Names:")
            if st.session_state.detected_objects_list:
                for obj in st.session_state.detected_objects_list:
                    st.markdown(f"- **{obj.upper()}**")
            else:
                st.write("*No identifiable items detected.*")
        else:
            st.warning("No detector initialized yet. Please use the Sidebar options.")