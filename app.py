import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'exit' not in st.session_state:
    st.session_state.exit = False

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO('besty.pt')
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        logger.error(f"Model loading error: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Define video processor for WebRTC
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"Error processing frame: {e}")
            return frame

# Streamlit UI layout
st.title("YOLO Object Detection with Webcam")
st.write("Real-time object detection using YOLO model from webcam feed.")
st.write("Please allow camera access when prompted by your browser.")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Webcam")
with col2:
    stop_button = st.button("Stop Webcam")
exit_button = st.button("Exit")

# Handle button actions
if start_button:
    st.session_state.running = True
    st.session_state.exit = False
if stop_button or exit_button:
    st.session_state.running = False
if exit_button:
    st.session_state.exit = True

# WebRTC streamer
if st.session_state.running and not st.session_state.exit:
    try:
        webrtc_streamer(
            key="yolo-webcam",
            video_processor_factory=YOLOProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={"video": True, "audio": False}
        )
    except Exception as e:
        st.error(f"WebRTC error: {e}")
        logger.error(f"WebRTC error: {e}")

if st.session_state.exit:
    st.write("Application terminated.")
    st.stop()