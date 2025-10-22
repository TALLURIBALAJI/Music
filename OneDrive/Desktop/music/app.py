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

# Mapping emotions to music files - Each emotion plays its corresponding MP3
EMOTION_MUSIC_FILES = {
    "happy": "musics/happy.mp3",
    "sad": "musics/sad.mp3",
    "angry": "musics/angry.mp3",
    "fear": "musics/fear.mp3",
    "surprise": "musics/Surprise.mp3",
    "neutral": "musics/Neutral.mp3",
    "disgust": "musics/Disgusting.mp3"
}

# Load emotion detection model once
@st.cache_resource
def load_emotion_model():
    return FER(mtcnn=False)

# Global variables for emotion detection
class EmotionState:
    def __init__(self):
        self.current_emotion = "neutral"
        self.last_emotion = "neutral"
        self.last_change_time = time.time()
        self.lock = threading.Lock()
        
emotion_state = EmotionState()

# Video processor for live streaming
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion_detector = load_emotion_model()
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect emotion every 10th frame for performance
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            try:
                # Resize for faster processing
                small = cv2.resize(img, (320, 240))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                # Detect emotion
                result = self.emotion_detector.detect_emotions(rgb)
                if result and len(result) > 0:
                    emotions = result[0]['emotions']
                    detected_emotion = max(emotions, key=emotions.get)
                    
                    # Update global state - Change song IMMEDIATELY when emotion changes
                    with emotion_state.lock:
                        emotion_state.current_emotion = detected_emotion
                        
                        # Change song immediately if emotion is different
                        if detected_emotion != emotion_state.last_emotion:
                            emotion_state.last_emotion = detected_emotion
                            emotion_state.last_change_time = time.time()
            except:
                pass
        
        # Draw emotion and song text on frame
        with emotion_state.lock:
            emotion_text = emotion_state.current_emotion.upper()
            playing_emotion = emotion_state.last_emotion.upper()
        
        # Draw background rectangles for better text visibility
        cv2.rectangle(img, (5, 5), (500, 40), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 45), (500, 80), (0, 0, 0), -1)
        
        # Draw emotion text
        cv2.putText(img, f"Emotion: {emotion_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw playing song text
        cv2.putText(img, f"Playing: {playing_emotion} Music", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Get music file bytes
def get_music_for_emotion(emotion):
    music_file = EMOTION_MUSIC_FILES.get(emotion, EMOTION_MUSIC_FILES["neutral"])
    if os.path.exists(music_file):
        with open(music_file, "rb") as f:
            return f.read()
    return None

# Page config
st.set_page_config(page_title="AI Music Composer", layout="centered")

# Hide refresh icon and Streamlit menu
hide_streamlit_style = """
<style>
button[title="View fullscreen"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stActionButton {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Title
st.title("🎵 AI Music Composer")
st.markdown("### Live Emotion Detection & Music Player")

# Initialize session state
if "last_played_emotion" not in st.session_state:
    st.session_state.last_played_emotion = None  # No default song
if "music_playing" not in st.session_state:
    st.session_state.music_playing = False  # Don't play until emotion is detected
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0
if "first_emotion_detected" not in st.session_state:
    st.session_state.first_emotion_detected = False

# Sidebar controls
st.sidebar.title("🎮 Controls")
stop_music = st.sidebar.button("⏹️ Stop Music")

if stop_music:
    st.session_state.music_playing = False
    st.session_state.audio_key += 1
    st.rerun()

# Main layout with columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("#### 📹 Live Camera")
    
    # WebRTC streamer for LIVE video
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Check for emotion changes and play music IMMEDIATELY
    with emotion_state.lock:
        current_emotion = emotion_state.last_emotion
    
    # Only play music when emotion is detected (not neutral at start)
    if current_emotion and current_emotion != st.session_state.last_played_emotion:
        st.session_state.last_played_emotion = current_emotion
        st.session_state.music_playing = True
        st.session_state.audio_key += 1
        st.session_state.first_emotion_detected = True
    
    # Music player - Play song matching detected emotion
    if st.session_state.music_playing and current_emotion and st.session_state.first_emotion_detected:
        audio_bytes = get_music_for_emotion(current_emotion)
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Emotion to emoji mapping
            emotion_emojis = {
                "happy": "😊",
                "sad": "😢",
                "angry": "😠",
                "fear": "😨",
                "surprise": "😲",
                "neutral": "😐",
                "disgust": "🤢"
            }
            
            emoji = emotion_emojis.get(current_emotion, "🎵")
            
            audio_html = f"""
            <audio id="audio_{st.session_state.audio_key}" autoplay loop controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.success(f"{emoji} **Now Playing:** {current_emotion.upper()} music → `musics/{current_emotion}.mp3`")
    
    # Download button
    if current_emotion and current_emotion in EMOTION_MUSIC_FILES:
        music_file = EMOTION_MUSIC_FILES[current_emotion]
        if os.path.exists(music_file):
            with open(music_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Current Music",
                    data=f,
                    file_name=f"{current_emotion}_music.mp3",
                    mime="audio/mp3"
                )

st.markdown("---")
st.info("💡 **Tip:** The camera is LIVE! Emotions are detected automatically every second and music plays based on your emotion.")
