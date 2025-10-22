import streamlit as st
import cv2
import numpy as np
import pretty_midi
import tempfile
import pygame
import time
import os
import base64
from io import BytesIO
import wave
from fer.fer import FER

# Modular functions: detect_emotion, generate_midi_for_emotion, midi_to_wav, play_sound

EMOTION_MAP = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'neutral': 'neutral',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust'
}

# Map emotions to music files
EMOTION_MUSIC_FILES = {
    'happy': 'musics/happy.mp3',
    'sad': 'musics/sad.mp3',
    'angry': 'musics/angry.mp3',
    'neutral': 'musics/Neutral.mp3',
    'surprise': 'musics/Surprise.mp3',
    'fear': 'musics/fear.mp3',
    'disgust': 'musics/Disgusting.mp3'  # Fallback to angry for disgust (no disgust.mp3 found)
}

st.set_page_config(page_title="AI Music Composer from Face Emotions", layout='wide')

# Emotion detection using FER (Facial Expression Recognition) - accurate CNN model
@st.cache_resource
def load_emotion_model():
    """Load pre-trained FER emotion detection model (accurate CNN-based)."""
    try:
        # FER with default OpenCV face detector (much faster than MTCNN)
        emotion_detector = FER(mtcnn=False)  # Use OpenCV Haar Cascade for speed
        return emotion_detector
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None

def detect_emotion(frame_bgr):
    """
    Detect emotion using FER (Facial Expression Recognition) library.
    Returns one of: happy, sad, angry, neutral, surprise, fear, disgust
    Uses deep learning CNN model trained on FER2013 dataset.
    Optimized for speed with frame resizing.
    """
    try:
        emotion_detector = load_emotion_model()
        if emotion_detector is None:
            return 'neutral'
        
        # Resize frame to very small size for maximum speed
        height, width = frame_bgr.shape[:2]
        scale = 240 / height  # Scale to 240p height for ultra-fast processing
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_small = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        # Convert BGR to RGB for FER
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Detect emotions
        result = emotion_detector.detect_emotions(frame_rgb)
        
        if not result or len(result) == 0:
            return 'neutral'
        
        # Get the first detected face's emotions
        emotions = result[0]['emotions']
        
        # Find the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        
        # Map FER emotions to our 7 emotions
        emotion_map = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral',
            'surprise': 'surprise',
            'fear': 'fear',
            'disgust': 'disgust'
        }
        
        return emotion_map.get(dominant_emotion, 'neutral')
    
    except Exception as e:
        return 'neutral'

# Load pre-recorded music file for emotion
def get_music_for_emotion(emotion):
    """
    Load the pre-recorded MP3 file for the detected emotion.
    Returns the file path and audio bytes.
    """
    try:
        music_file = EMOTION_MUSIC_FILES.get(emotion, 'musics/Neutral.mp3')
        music_path = os.path.join(os.path.dirname(__file__), music_file)
        
        if not os.path.exists(music_path):
            st.error(f"Music file not found: {music_path}")
            return None, None
        
        # Read the audio file
        with open(music_path, 'rb') as f:
            audio_bytes = f.read()
        
        return music_path, audio_bytes
    except Exception as e:
        st.error(f"Error loading music: {e}")
        return None, None


def play_music_file(music_path):
    """Play MP3 file using pygame mixer with high quality settings."""
    try:
        pygame.mixer.quit()
        # Initialize with higher quality settings for clear audio
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(1.0)  # Full volume
        pygame.mixer.music.play()
    except Exception as e:
        st.error(f"Playback error: {e}")


# Music generation
def generate_midi_for_emotion(emotion, length_seconds=6, tempo=120):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Choose scale and note lengths for each emotion
    if emotion == 'happy':
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major
        bpm = 140
    elif emotion == 'sad':
        scale = [60, 62, 63, 65, 67, 68, 70, 72]  # C minor-ish
        bpm = 60
    elif emotion == 'angry':
        scale = [60, 61, 63, 64, 66, 67, 69, 70]  # dissonant
        bpm = 120
    elif emotion == 'surprise':
        scale = [60, 64, 67, 72, 76, 79]  # playful jumps
        bpm = 160
    elif emotion == 'fear':
        scale = [60, 63, 67, 70]  # minor triad, sparse
        bpm = 100
        # Fear: trembling, uncertain rhythm
    elif emotion == 'disgust':
        scale = [60, 61, 63, 65, 68]  # chromatic, off-scale notes
        bpm = 80
    else:  # neutral
        scale = [60, 62, 64, 65, 67, 69, 71]  # neutral
        bpm = 100

    # Create a simple melody
    seconds_per_beat = 60.0 / bpm
    current_time = 0.0
    notes = []
    rng = np.random.RandomState(seed=int(time.time()) % (2**32 - 1))

    while current_time < length_seconds:
        if emotion == 'angry':
            duration = seconds_per_beat * rng.choice([0.25, 0.5, 1.0])
            velocity = rng.randint(80, 127)
            pitch = rng.choice(scale) + rng.choice([0, 12, 24]) + rng.choice([-1, 1])
        elif emotion == 'surprise':
            duration = seconds_per_beat * rng.choice([0.25, 0.5])
            velocity = rng.randint(60, 110)
            pitch = rng.choice(scale) + rng.choice([0, 12])
        elif emotion == 'fear':
            # Trembling effect: very short notes with variations
            duration = seconds_per_beat * rng.choice([0.125, 0.25, 0.5])
            velocity = rng.randint(40, 70)
            pitch = rng.choice(scale) + rng.choice([-2, -1, 0, 1, 2])
        elif emotion == 'disgust':
            # Unpleasant: chromatic, off-note, irregular rhythm
            duration = seconds_per_beat * rng.choice([0.25, 0.75, 1.0])
            velocity = rng.randint(50, 90)
            pitch = rng.choice(scale) + rng.choice([0, 1, 2, -1])
        elif emotion == 'happy':
            duration = seconds_per_beat * rng.choice([0.25, 0.5, 1.0])
            velocity = rng.randint(70, 110)
            pitch = rng.choice(scale) + 0
        elif emotion == 'sad':
            duration = seconds_per_beat * rng.choice([0.5, 1.0, 1.5])
            velocity = rng.randint(50, 80)
            pitch = rng.choice(scale) - 12
        else:  # neutral
            duration = seconds_per_beat * rng.choice([0.5, 1.0])
            velocity = rng.randint(55, 95)
            pitch = rng.choice(scale)

        start = current_time
        end = current_time + duration
        note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=start, end=end)
        instrument.notes.append(note)
        current_time = end

    pm.instruments.append(instrument)
    pm.resolution = 960
    return pm

# Render MIDI to WAV using fluidsynth is not available; use pygame midi synth via writing raw audio
# We'll synthesize a simple sine wave for each midi note and mix — keep it simple to avoid external dependencies.

def midi_to_wav(pretty_midi_obj, sample_rate=44100):
    duration = max([note.end for inst in pretty_midi_obj.instruments for note in inst.notes]) if pretty_midi_obj.instruments else 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    for inst in pretty_midi_obj.instruments:
        for note in inst.notes:
            freq = pretty_midi.note_number_to_hz(note.pitch)
            start_i = int(note.start * sample_rate)
            end_i = min(int(note.end * sample_rate), len(t))
            tt = np.linspace(0, note.end - note.start, end_i - start_i, endpoint=False)
            envelope = np.linspace(1.0, 0.01, len(tt))
            wave = 0.1 * (note.velocity / 127.0) * np.sin(2 * np.pi * freq * tt) * envelope
            audio[start_i:end_i] += wave

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    audio_int16 = np.int16(audio * 32767)
    return sample_rate, audio_int16


def play_wav_data(sample_rate, audio_int16):
    """Play audio using pygame mixer with correct mono format."""
    try:
        pygame.mixer.quit()
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=2048)
        sound = pygame.sndarray.make_sound(audio_int16)
        sound.play()
        return sound
    except Exception as e:
        st.error(f"Playback error: {e}")
        return None


# Streamlit UI
st.title("AI Music Composer from Face Emotions")

st.sidebar.markdown("## Controls")
run_camera = st.sidebar.button("Start Camera")
stop_camera = st.sidebar.button("Stop Camera")
stop_music = st.sidebar.button("Stop Music")

# Initialize placeholders in session state (persist across reruns)
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = st.empty()
if 'emotion_placeholder' not in st.session_state:
    st.session_state.emotion_placeholder = st.empty()
if 'play_placeholder' not in st.session_state:
    st.session_state.play_placeholder = st.empty()
if 'download_placeholder' not in st.session_state:
    st.session_state.download_placeholder = st.empty()

# Use placeholders from session state
frame_placeholder = st.session_state.frame_placeholder
emotion_placeholder = st.session_state.emotion_placeholder
play_placeholder = st.session_state.play_placeholder
download_placeholder = st.session_state.download_placeholder

# Webcam loop
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = 'neutral'
if 'emotion_freeze_until' not in st.session_state:
    st.session_state.emotion_freeze_until = 0
if 'frozen_emotion' not in st.session_state:
    st.session_state.frozen_emotion = 'neutral'
if 'current_audio_bytes' not in st.session_state:
    st.session_state.current_audio_bytes = None
if 'current_audio_html' not in st.session_state:
    st.session_state.current_audio_html = ""
if 'audio_key' not in st.session_state:
    st.session_state.audio_key = 0
if 'audio_dirty' not in st.session_state:
    st.session_state.audio_dirty = False
if 'download_key' not in st.session_state:
    st.session_state.download_key = "download_0"
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'frame_skip_counter' not in st.session_state:
    st.session_state.frame_skip_counter = 0
if 'cached_emotion' not in st.session_state:
    st.session_state.cached_emotion = 'neutral'

if run_camera:
    st.session_state.running = True
if stop_camera:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
if stop_music:
    play_placeholder.empty()
    download_placeholder.empty()
    st.session_state.current_audio_bytes = None
    st.session_state.current_audio_html = ""
    st.session_state.audio_dirty = False
    st.session_state.download_key = f"download_stop_{int(time.time()*1000)}"
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

if st.session_state.running:
    # Open camera if not already open
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("Could not open webcam")
            st.session_state.running = False
            st.session_state.cap = None
    
    if st.session_state.cap is not None:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Failed to read frame from webcam")
            st.session_state.cap.release()
            st.session_state.cap = None
        else:
            # Detect emotion every 5th frame for better balance of speed and responsiveness
            st.session_state.frame_skip_counter += 1
            
            # Always detect on first frame, then every 5th frame
            if st.session_state.frame_skip_counter == 1 or st.session_state.frame_skip_counter % 5 == 0:
                detected = detect_emotion(frame)
                if detected:  # Only update if detection successful
                    st.session_state.cached_emotion = detected
            
            emotion = st.session_state.cached_emotion

            current_time = time.time()
            is_frozen = current_time < st.session_state.emotion_freeze_until

            if is_frozen:
                display_emotion = st.session_state.frozen_emotion
                time_remaining = max(int(st.session_state.emotion_freeze_until - current_time), 0)
            else:
                display_emotion = emotion
                time_remaining = 0

            # Display frame directly without face rectangle for maximum speed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)

            if is_frozen:
                emotion_placeholder.markdown(
                    f"### 😊 Detected emotion: **{display_emotion.upper()}** 🔒 (Locked for {time_remaining}s)"
                )
            else:
                emotion_placeholder.markdown(f"### 😊 Current emotion: **{display_emotion.upper()}**")
            
            # Debug: Show frame counter
            st.sidebar.text(f"Frame: {st.session_state.frame_skip_counter}")

            should_play_music = False
            if not is_frozen and display_emotion != st.session_state.last_emotion:
                should_play_music = True
                st.session_state.last_emotion = display_emotion
                st.session_state.frozen_emotion = display_emotion
                st.session_state.emotion_freeze_until = current_time + 25

            if should_play_music:
                music_path, audio_file_bytes = get_music_for_emotion(display_emotion)
                if music_path and audio_file_bytes:
                    try:
                        try:
                            pygame.mixer.music.stop()
                        except Exception:
                            pass

                        st.session_state.current_audio_bytes = audio_file_bytes
                        st.session_state.audio_key += 1
                        st.session_state.download_key = f"download_{st.session_state.audio_key}_{int(time.time()*1000)}"
                        encoded_audio = base64.b64encode(audio_file_bytes).decode('utf-8')
                        st.session_state.current_audio_html = (
                            f'<audio id="emotion-audio" data-version="{st.session_state.audio_key}" '
                            'controls autoplay loop style="width:100%;">'
                            f'<source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">'
                            "Your browser does not support the audio element."
                            "</audio>"
                        )
                        st.session_state.audio_dirty = True
                    except Exception as e:
                        st.error(f"Audio processing error: {e}")

            if st.session_state.current_audio_bytes is not None:
                if st.session_state.audio_dirty:
                    play_placeholder.empty()
                    download_placeholder.empty()
                    st.session_state.audio_dirty = False

                play_placeholder.markdown(
                    st.session_state.current_audio_html,
                    unsafe_allow_html=True
                )

                download_placeholder.download_button(
                    label=f'💾 Download {st.session_state.last_emotion.capitalize()} Music',
                    data=st.session_state.current_audio_bytes,
                    file_name=f'{st.session_state.last_emotion}_music.mp3',
                    mime='audio/mp3',
                    key=st.session_state.download_key
                )
            
            # Minimal delay for smooth camera (30ms = ~33 FPS)
            time.sleep(0.03)
            st.rerun()
else:
    # Release camera when stopped
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.info("Camera is stopped. Click 'Start Camera' in the sidebar to begin.")

st.markdown("---")
st.markdown("Notes: Uses DeepFace for emotion detection. The first run may download models.")

# Footer
st.caption('Built with OpenCV, DeepFace, PrettyMIDI, and Pygame')
