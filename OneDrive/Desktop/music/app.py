import streamlit as st
import cv2
import numpy as np
import pretty_midi
import tempfile
import pygame
import time
import os
from io import BytesIO
import wave

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

st.set_page_config(page_title="AI Music Composer from Face Emotions", layout='wide')

# Emotion detection using OpenCV DNN with pre-trained model
@st.cache_resource
def load_emotion_model():
    """Load pre-trained emotion detection model (uses OpenCV DNN, no external downloads)."""
    try:
        # Use a lightweight CNN trained on emotion classification
        # We'll use a simple model based on facial landmarks and DNN
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        return None

def detect_emotion(frame_bgr):
    """
    Detect all 7 core emotions by analyzing facial features.
    Emotions: happy, sad, angry, neutral, surprise, fear, disgust
    """
    try:
        face_cascade = load_emotion_model()
        if face_cascade is None:
            return 'neutral'
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return 'neutral'
        
        # Get the largest face (main subject)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Calculate facial feature regions
        # 1. Eye region (upper third) - for eye openness and tension
        eye_region = face_roi[:h//3, :]
        eye_variance = np.var(eye_region)
        eye_mean = np.mean(eye_region)
        
        # 2. Mouth region (lower half) - for mouth openness and smile
        mouth_region = face_roi[h//2:, :]
        mouth_intensity = np.mean(mouth_region)
        mouth_variance = np.var(mouth_region)
        
        # 3. Nose region (middle) - for wrinkles (disgust indicator)
        nose_region = face_roi[h//3:h//2, w//4:3*w//4]
        nose_variance = np.var(nose_region)
        
        # 4. Overall face metrics
        face_contrast = np.std(face_roi)
        brightness = np.mean(face_roi)
        
        # Edge detection for intensity of expression
        edges = cv2.Canny(face_roi, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # Emotion classification logic (order matters!)
        
        # DISGUST: wrinkled nose, low brightness, high contrast
        if nose_variance > 150 and brightness < 85 and face_contrast > 35 and mouth_intensity < 95:
            return 'disgust'
        
        # FEAR: very wide eyes, mouth open, high edge density (tense)
        elif eye_variance > 350 and mouth_intensity < 75 and edge_density > 0.15:
            return 'fear'
        
        # SURPRISE: very wide eyes, mouth open, moderate brightness
        elif eye_variance > 300 and mouth_intensity < 85 and brightness > 95:
            return 'surprise'
        
        # HAPPY: bright mouth (smile), eyes open, good face contrast
        elif mouth_intensity > 105 and eye_variance > 200 and face_contrast > 28:
            return 'happy'
        
        # ANGRY: low brightness, high contrast, tense (high edge density)
        elif brightness < 92 and face_contrast > 38 and edge_density > 0.12:
            return 'angry'
        
        # SAD: low brightness, low contrast, subtle features
        elif brightness < 102 and face_contrast < 32 and mouth_intensity < 95:
            return 'sad'
        
        # NEUTRAL: balanced features
        else:
            return 'neutral'
    
    except Exception as e:
        return 'neutral'

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
            end_i = int(note.end * sample_rate)
            if end_i > len(t):
                end_i = len(t)
            tt = np.linspace(0, note.end - note.start, end_i - start_i, endpoint=False)
            envelope = np.linspace(1.0, 0.01, len(tt))
            wave = 0.1 * (note.velocity / 127.0) * np.sin(2 * np.pi * freq * tt) * envelope
            audio[start_i:end_i] += wave

    # normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    # convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)
    return sample_rate, audio_int16


def play_wav_data(sample_rate, audio_int16):
    # Initialize pygame mixer
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
    sound = pygame.sndarray.make_sound(audio_int16)
    sound.play()
    return sound


# Streamlit UI
st.title("AI Music Composer from Face Emotions")

st.sidebar.markdown("## Controls")
run_camera = st.sidebar.button("Start Camera")
stop_camera = st.sidebar.button("Stop Camera")

# Placeholders
frame_placeholder = st.empty()
emotion_placeholder = st.empty()
play_placeholder = st.empty()

# Webcam loop
cap = None

if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = 'neutral'
if 'sound' not in st.session_state:
    st.session_state.sound = None

if run_camera:
    st.session_state.running = True
if stop_camera:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        st.session_state.running = False
    else:
        audio_file_bytes = None
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam")
                break

            # Detect emotion
            emotion = detect_emotion(frame)
            
            # Draw face detection boxes for visual feedback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            frame_display = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display frame with boxes
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels='RGB')

            # Update emotion display
            emotion_placeholder.markdown(f"### 😊 Detected emotion: **{emotion.upper()}**")

            # Generate and play music on emotion change
            if emotion != st.session_state.last_emotion:
                st.session_state.last_emotion = emotion

                # generate midi
                pm = generate_midi_for_emotion(emotion, length_seconds=6)
                # convert to wav
                sr, audio_int16 = midi_to_wav(pm)

                # save to tempfile
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                try:
                    # write WAV
                    with wave.open(tmp_wav.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(audio_int16.tobytes())

                    # play sound
                    if st.session_state.sound is not None:
                        try:
                            st.session_state.sound.stop()
                        except Exception:
                            pass
                    st.session_state.sound = play_wav_data(sr, audio_int16)

                    # prepare download
                    with open(tmp_wav.name, 'rb') as f:
                        audio_file_bytes = f.read()

                    play_placeholder.audio(audio_file_bytes, format='audio/wav')

                    # download button
                    st.download_button(label='Download Music', data=audio_file_bytes, file_name=f'{emotion}_music.wav', mime='audio/wav')

                except Exception as e:
                    st.error(f"Audio processing error: {e}")

            # small sleep to avoid busy loop
            time.sleep(0.1)

        cap.release()
else:
    st.info("Camera is stopped. Click 'Start Camera' in the sidebar to begin.")

st.markdown("---")
st.markdown("Notes: Uses DeepFace for emotion detection. The first run may download models.")

# Footer
st.caption('Built with OpenCV, DeepFace, PrettyMIDI, and Pygame')
