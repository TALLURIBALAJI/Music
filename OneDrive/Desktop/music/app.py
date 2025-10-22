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

# Emotion configuration - inspired by emo-melodies emotional design system
EMOTION_CONFIG = {
    "happy": {
        "emoji": "😊",
        "label": "Happy",
        "color": "#FFD700",  # Gold
        "music_file": "musics/happy.mp3"
    },
    "sad": {
        "emoji": "😢",
        "label": "Sad",
        "color": "#4169E1",  # Royal Blue
        "music_file": "musics/sad.mp3"
    },
    "angry": {
        "emoji": "😠",
        "label": "Angry",
        "color": "#DC143C",  # Crimson
        "music_file": "musics/angry.mp3"
    },
    "fear": {
        "emoji": "😨",
        "label": "Fear",
        "color": "#9370DB",  # Medium Purple
        "music_file": "musics/fear.mp3"
    },
    "surprise": {
        "emoji": "😮",
        "label": "Surprise",
        "color": "#FF8C00",  # Dark Orange
        "music_file": "musics/Surprise.mp3"
    },
    "neutral": {
        "emoji": "😐",
        "label": "Neutral",
        "color": "#808080",  # Gray
        "music_file": "musics/Neutral.mp3"
    },
    "disgust": {
        "emoji": "🤢",
        "label": "Disgust",
        "color": "#32CD32",  # Lime Green
        "music_file": "musics/Disgusting.mp3"
    }
}

# Load emotion detection model once
@st.cache_resource
def load_emotion_model():
    return FER(mtcnn=False)

# Global variables for emotion detection
class EmotionState:
    def __init__(self):
        self.current_emotion = None  # No default emotion
        self.last_emotion = None  # No default song
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
        
        # Detect emotion every 5th frame for faster real-time detection (improved from 10)
        self.frame_count += 1
        if self.frame_count % 5 == 0:
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
            emotion_text = emotion_state.current_emotion.upper() if emotion_state.current_emotion else "DETECTING..."
            playing_emotion = emotion_state.last_emotion.upper() if emotion_state.last_emotion else "NONE"
        
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

# Get music file path for emotion
def get_music_file(emotion):
    """Get music file path for given emotion"""
    if not emotion or emotion not in EMOTION_CONFIG:
        return None
    return EMOTION_CONFIG[emotion]["music_file"]

# Page config - inspired by emo-melodies modern UI
st.set_page_config(
    page_title="AI Music Composer - Transform Emotions into Melodies",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI (inspired by emo-melodies design system)
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stActionButton {display: none;}
    button[title="View fullscreen"] {display: none;}
    
    /* Gradient text for title */
    .gradient-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Smooth animations */
    .stMarkdown, .stAudio {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Better button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stDownloadButton button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Title with gradient effect
st.markdown('<h1 class="gradient-title">🎵 AI Music Composer</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p style="font-size: 18px; color: #666;">
        ✨ <em>Transform emotions into melodies with AI</em> ✨
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for tracking
if "last_played_emotion" not in st.session_state:
    st.session_state.last_played_emotion = None
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0
if "_last_refresh" not in st.session_state:
    st.session_state._last_refresh = 0

# Sidebar controls with better UI
with st.sidebar:
    st.markdown("### 🎮 Controls")
    st.markdown("---")
    
    # Stop music button
    if st.button("⏹️ Stop Music", use_container_width=True):
        st.session_state.last_played_emotion = None
        st.session_state.audio_key += 1
        st.rerun()
    
    st.markdown("---")
    
    # Stats and info
    with emotion_state.lock:
        current = emotion_state.current_emotion
    
    st.markdown("### 📊 Status")
    if current:
        config = EMOTION_CONFIG.get(current, {"emoji": "❓", "label": "Unknown"})
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 48px;">{config['emoji']}</div>
            <p style="margin: 5px 0; font-weight: 600;">{config['label']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👀 No emotion detected")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 12px; color: #888; text-align: center;">
        <p>🎭 7 emotions supported</p>
        <p>🎵 Real-time music sync</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
st.markdown("---")

# Main layout - centered single column for better focus
st.markdown("### 📹 Live Camera Feed")
st.markdown("""
<div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
    <p style="margin: 0; color: #666; font-size: 14px;">
        🎥 Camera is LIVE • Emotions detected automatically • Music syncs in real-time
    </p>
</div>
""", unsafe_allow_html=True)

# WebRTC streamer for LIVE video - centered and prominent
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Get current emotion from global state (real-time from video processor)
with emotion_state.lock:
    current_emotion = emotion_state.last_emotion

# Check if emotion changed - update audio key to force audio refresh
if current_emotion != st.session_state.last_played_emotion:
    st.session_state.last_played_emotion = current_emotion
    st.session_state.audio_key += 1
    # Force rerun to load new audio immediately
    st.rerun()

# Music player - Display detected emotion with styled UI
st.markdown("---")
st.markdown("### 🎭 Detected Emotion & Music Player")

if current_emotion and current_emotion in EMOTION_CONFIG:
    config = EMOTION_CONFIG[current_emotion]
    
    # Styled emotion display
    emotion_html = f"""
    <div style="background: linear-gradient(135deg, {config['color']}22, {config['color']}44); 
                padding: 30px; 
                border-radius: 20px; 
                text-align: center;
                border: 2px solid {config['color']}88;
                margin: 20px 0;">
        <div style="font-size: 80px; margin-bottom: 10px;">{config['emoji']}</div>
        <h2 style="color: {config['color']}; margin: 10px 0; font-size: 36px;">{config['label']}</h2>
        <p style="color: #666; font-size: 16px;">Music playing to match your emotion</p>
    </div>
    """
    st.markdown(emotion_html, unsafe_allow_html=True)
    
    # Get the music file path
    music_file = get_music_file(current_emotion)
    if music_file and os.path.exists(music_file):
        # Read audio file
        with open(music_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        st.markdown(f"""
            <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <p style="text-align: center; margin: 0; font-size: 18px;">
                    🎵 <strong>Now Playing: {config['label']} Music</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visible audio player that changes when emotion changes
        st.audio(audio_bytes, format='audio/mp3', start_time=0, key=f"audio_{st.session_state.audio_key}")
        
        # Add autoplay HTML audio (hidden, loops continuously)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        autoplay_html = f"""
        <audio id="emotion_audio_{st.session_state.audio_key}" autoplay loop>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            // Auto-play the audio
            var audio = document.getElementById('emotion_audio_{st.session_state.audio_key}');
            if (audio) {{
                audio.volume = 0.7;
                audio.play().catch(function(error) {{
                    console.log("Autoplay prevented:", error);
                }});
            }}
        </script>
        """
        st.markdown(autoplay_html, unsafe_allow_html=True)
else:
    # No emotion detected yet
    st.markdown("""
    <div style="background: #f8f9fa; 
                padding: 40px; 
                border-radius: 20px; 
                text-align: center;
                border: 2px dashed #ccc;">
        <div style="font-size: 60px; margin-bottom: 15px;">📸</div>
        <p style="color: #666; font-size: 18px;">No emotion detected yet</p>
        <p style="color: #999; font-size: 14px;">Face the camera to begin emotion detection</p>
    </div>
    """, unsafe_allow_html=True)

# Download button
if current_emotion and current_emotion in EMOTION_CONFIG:
    music_file = get_music_file(current_emotion)
    if music_file and os.path.exists(music_file):
        with open(music_file, "rb") as f:
            config = EMOTION_CONFIG[current_emotion]
            st.download_button(
                label=f"⬇️ Download {config['label']} Music",
                data=f,
                file_name=f"{current_emotion}_music.mp3",
                mime="audio/mp3",
                use_container_width=True
            )

st.markdown("---")

# Info section with styled design
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea22, #764ba244); 
            padding: 25px; 
            border-radius: 15px;
            border-left: 5px solid #667eea;">
    <h3 style="margin-top: 0; color: #667eea;">💡 How It Works</h3>
    <ol style="color: #555; line-height: 1.8;">
        <li>📹 <strong>Allow camera access</strong> when prompted</li>
        <li>🎭 <strong>Face the camera</strong> - emotions are detected automatically every second</li>
        <li>🎵 <strong>Music plays automatically</strong> based on your detected emotion</li>
        <li>🔄 <strong>Emotion changes?</strong> Music changes with it!</li>
    </ol>
    <p style="margin-bottom: 0; color: #888; font-size: 14px;">
        ✨ <em>Powered by FER emotion detection & live video streaming</em>
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 14px; padding: 20px 0;">
    <p>Built with ❤️ using Streamlit • Every face has a melody 🎭🎵</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh every 1 second to check for emotion changes in real-time
if st.session_state.get("_last_refresh", 0) + 1 < time.time():
    st.session_state._last_refresh = time.time()
    time.sleep(0.05)
    st.rerun()
