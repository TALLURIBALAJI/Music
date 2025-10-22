import streamlit as st
import av
import time
import os
import base64
from fer.fer import FER
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import threading

# Page config
st.set_page_config(page_title="AI Music Composer from Face Emotions", layout='wide')

# Emotion to music file mapping
import streamlit as st
import av
import cv2
import numpy as np
import time
import os
import base64
from fer.fer import FER
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import threading

# Mapping emotions to music files
EMOTION_MUSIC_FILES = {
    'angry': 'musics/angry.mp3',
    'disgust': 'musics/angry.mp3',  # Using angry as fallback
    'fear': 'musics/fear.mp3',
    'happy': 'musics/happy.mp3',
    'sad': 'musics/sad.mp3',
    'surprise': 'musics/Surprise.mp3',
    'neutral': 'musics/Neutral.mp3'
}

# Load emotion detection model (cached)
@st.cache_resource
def load_emotion_model():
    """Load FER model with fast OpenCV face detection"""
    try:
        return FER(mtcnn=False)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Detect emotion from frame
def detect_emotion(frame):
    """Detect emotion using FER with optimized frame size"""
    try:
        model = load_emotion_model()
        if model is None:
            return 'neutral'
        
        # Resize to 240p for speed
        h, w = frame.shape[:2]
        scale = 240 / h
        small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        result = model.detect_emotions(rgb)
        if result and len(result) > 0:
            emotions = result[0]['emotions']
            return max(emotions, key=emotions.get)
        return 'neutral'
    except:
        return 'neutral'

# Get music file
def get_music_for_emotion(emotion):
    """Load MP3 file for emotion"""
    try:
        path = EMOTION_MUSIC_FILES.get(emotion.lower(), EMOTION_MUSIC_FILES['neutral'])
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
    except Exception as e:
        st.error(f"Music loading error: {e}")
    return None

# UI
st.title("🎵 AI Music Composer from Face Emotions")

# Sidebar controls
st.sidebar.markdown("## Controls")
run_camera = st.sidebar.button("▶️ Start Camera")
stop_camera = st.sidebar.button("⏹️ Stop Camera")
refresh_button = st.sidebar.button("🔄 Refresh Frame")
stop_music = st.sidebar.button("🔇 Stop Music")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = 'neutral'
if 'cached_emotion' not in st.session_state:
    st.session_state.cached_emotion = 'neutral'
if 'freeze_until' not in st.session_state:
    st.session_state.freeze_until = 0
if 'frozen_emotion' not in st.session_state:
    st.session_state.frozen_emotion = 'neutral'
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'audio_html' not in st.session_state:
    st.session_state.audio_html = ""
if 'audio_key' not in st.session_state:
    st.session_state.audio_key = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_rerun_time' not in st.session_state:
    st.session_state.last_rerun_time = 0

# Create persistent placeholders
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = st.empty()
if 'emotion_placeholder' not in st.session_state:
    st.session_state.emotion_placeholder = st.empty()
if 'audio_placeholder' not in st.session_state:
    st.session_state.audio_placeholder = st.empty()
if 'download_placeholder' not in st.session_state:
    st.session_state.download_placeholder = st.empty()

# Create columns for smaller camera display
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    frame_placeholder = st.session_state.frame_placeholder
    emotion_placeholder = st.session_state.emotion_placeholder

audio_placeholder = st.session_state.audio_placeholder
download_placeholder = st.session_state.download_placeholder

# Handle button clicks
if run_camera:
    st.session_state.running = True
    st.session_state.frame_count = 0

if stop_camera:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

if stop_music:
    audio_placeholder.empty()
    download_placeholder.empty()
    st.session_state.audio_bytes = None
    st.session_state.audio_html = ""

# Main camera loop
if st.session_state.running:
    # Open camera
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("❌ Cannot open webcam")
            st.session_state.running = False
    
    if st.session_state.cap:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame")
            st.session_state.cap.release()
            st.session_state.cap = None
        else:
            st.session_state.frame_count += 1
            
            # Detect emotion less frequently to reduce load
            if st.session_state.frame_count % 3 == 0:  # Every 3rd refresh
                detected = detect_emotion(frame)
                if detected:
                    st.session_state.cached_emotion = detected
            
            emotion = st.session_state.cached_emotion
            current_time = time.time()
            
            # Check freeze timer
            is_frozen = current_time < st.session_state.freeze_until
            if is_frozen:
                display_emotion = st.session_state.frozen_emotion
                time_left = int(st.session_state.freeze_until - current_time)
            else:
                display_emotion = emotion
                time_left = 0
            
            # Display frame (resized for smaller display)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize display frame to 640x480 for smaller size
            frame_display = cv2.resize(frame_rgb, (640, 480))
            frame_placeholder.image(frame_display, channels='RGB', use_container_width=True)
            
            # Display emotion
            if is_frozen:
                emotion_placeholder.markdown(f"### 😊 **{display_emotion.upper()}** 🔒 (Locked {time_left}s)")
            else:
                emotion_placeholder.markdown(f"### 😊 **{display_emotion.upper()}**")
            
            # Play music on emotion change
            if not is_frozen and display_emotion != st.session_state.last_emotion:
                st.session_state.last_emotion = display_emotion
                st.session_state.frozen_emotion = display_emotion
                st.session_state.freeze_until = current_time + 25
                
                # Load and play music
                audio_bytes = get_music_for_emotion(display_emotion)
                if audio_bytes:
                    st.session_state.audio_bytes = audio_bytes
                    st.session_state.audio_key += 1
                    encoded = base64.b64encode(audio_bytes).decode()
                    st.session_state.audio_html = f'''
                        <audio controls autoplay loop style="width:100%;">
                            <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
                        </audio>
                    '''
            
            # Display audio player
            if st.session_state.audio_bytes:
                audio_placeholder.markdown(st.session_state.audio_html, unsafe_allow_html=True)
                download_placeholder.download_button(
                    label=f"💾 Download {st.session_state.last_emotion.capitalize()} Music",
                    data=st.session_state.audio_bytes,
                    file_name=f"{st.session_state.last_emotion}_music.mp3",
                    mime="audio/mp3",
                    key=f"download_{st.session_state.audio_key}"
                )
            
            # Show instruction
            st.info("👆 Click '🔄 Refresh Frame' button in sidebar to update camera and detect emotion")
else:
    # Camera stopped
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.info("📷 Camera stopped. Click 'Start Camera' to begin.")

# Footer
st.markdown("---")
st.caption("Built with OpenCV, FER, and Streamlit")
