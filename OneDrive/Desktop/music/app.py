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
from typing import Optional, Tuple, Dict, Any

# Optional DeepFace import (robust fallback to heuristic if unavailable)
try:
    from deepface import DeepFace  # type: ignore
    HAS_DEEPFACE = True
except Exception:
    DeepFace = None  # type: ignore
    HAS_DEEPFACE = False


# ----------------------------
# Page config & global styling
# ----------------------------
st.set_page_config(
    page_title="AI Music Composer from Face Emotions",
    page_icon="🎵",
    layout="wide",
)

ACCENT = "#6A5ACD"  # Royal purple
EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

EMOTION_COLORS: Dict[str, str] = {
    "happy": "#FFD54F",
    "sad": "#64B5F6",
    "angry": "#E53935",
    "surprise": "#EA80FC",
    "fear": "#00897B",
    "disgust": "#7CB342",
    "neutral": "#BDBDBD",
}

EMOTION_BG_GRADIENTS: Dict[str, Tuple[str, str]] = {
    "happy": ("#FFF176", "#FFD54F"),
    "sad": ("#90CAF9", "#64B5F6"),
    "angry": ("#EF5350", "#E53935"),
    "surprise": ("#FF80AB", "#EA80FC"),
    "fear": ("#4DB6AC", "#00897B"),
    "disgust": ("#AED581", "#7CB342"),
    "neutral": ("#E0E0E0", "#BDBDBD"),
}

style_placeholder = st.empty()

def inject_base_css() -> None:
    style_placeholder.markdown(
        f"""
        <style>
            .app-title {{
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(90deg, {ACCENT}, #8B80F9, #9C27B0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .emotion-badge {{
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-weight: 700;
                background-color: rgba(0,0,0,0.08);
            }}
            .card {{
                background: rgba(255,255,255,0.66);
                border: 1px solid rgba(0,0,0,0.05);
                border-radius: 12px;
                padding: 1rem 1.25rem;
                box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            }}
            .subtle {{ opacity: 0.85; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_dynamic_background(emotion: str) -> None:
    a, b = EMOTION_BG_GRADIENTS.get(emotion, EMOTION_BG_GRADIENTS["neutral"])  # type: ignore
    style_placeholder.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(135deg, {a} 0%, {b} 55%);
                transition: background 400ms ease-in-out;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Emotion detection
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion_heuristic(frame_bgr: np.ndarray) -> str:
    try:
        face_cascade = load_face_cascade()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        if len(faces) == 0:
            return "neutral"
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y + h, x:x + w]

        eye_region = face_roi[: h // 3, :]
        eye_variance = float(np.var(eye_region))
        mouth_region = face_roi[h // 2 :, :]
        mouth_intensity = float(np.mean(mouth_region))
        mouth_variance = float(np.var(mouth_region))
        nose_region = face_roi[h // 3 : h // 2, w // 4 : 3 * w // 4]
        nose_variance = float(np.var(nose_region))
        face_contrast = float(np.std(face_roi))
        brightness = float(np.mean(face_roi))
        edges = cv2.Canny(face_roi, 100, 200)
        edge_density = float(np.mean(edges)) / 255.0

        if nose_variance > 150 and brightness < 85 and face_contrast > 35 and mouth_intensity < 95:
            return "disgust"
        elif eye_variance > 350 and mouth_intensity < 75 and edge_density > 0.15:
            return "fear"
        elif eye_variance > 300 and mouth_intensity < 85 and brightness > 95:
            return "surprise"
        elif mouth_intensity > 105 and eye_variance > 200 and face_contrast > 28:
            return "happy"
        elif brightness < 92 and face_contrast > 38 and edge_density > 0.12:
            return "angry"
        elif brightness < 102 and face_contrast < 32 and mouth_intensity < 95:
            return "sad"
        else:
            return "neutral"
    except Exception:
        return "neutral"


def detect_emotion_deepface(frame_bgr: np.ndarray) -> str:
    if not HAS_DEEPFACE:
        raise RuntimeError("DeepFace not available")
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res: Any = DeepFace.analyze(  # type: ignore
            rgb,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        dom = str(res.get("dominant_emotion", "neutral")).lower()
        return dom if dom in EMOTIONS else "neutral"
    except Exception:
        return detect_emotion_heuristic(frame_bgr)


def detect_emotion(frame_bgr: np.ndarray, prefer_deepface: bool = True) -> str:
    if prefer_deepface and HAS_DEEPFACE:
        return detect_emotion_deepface(frame_bgr)
    return detect_emotion_heuristic(frame_bgr)


# ----------------------------
# Music generation (PrettyMIDI)
# ----------------------------

def generate_midi_for_emotion(emotion: str, length_seconds: float = 6.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    if emotion == "happy":
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        bpm = 140
    elif emotion == "sad":
        scale = [60, 62, 63, 65, 67, 68, 70, 72]
        bpm = 60
    elif emotion == "angry":
        scale = [60, 61, 63, 64, 66, 67, 69, 70]
        bpm = 120
    elif emotion == "surprise":
        scale = [60, 64, 67, 72, 76, 79]
        bpm = 160
    elif emotion == "fear":
        scale = [60, 63, 67, 70]
        bpm = 100
    elif emotion == "disgust":
        scale = [60, 61, 63, 65, 68]
        bpm = 80
    else:
        scale = [60, 62, 64, 65, 67, 69, 71]
        bpm = 100

    seconds_per_beat = 60.0 / bpm
    current_time = 0.0
    rng = np.random.RandomState(seed=int(time.time()) % (2**32 - 1))

    while current_time < length_seconds:
        if emotion == "angry":
            duration = seconds_per_beat * rng.choice([0.25, 0.5, 1.0])
            velocity = int(rng.randint(95, 127))
            pitch = int(rng.choice(scale) + rng.choice([0, 12, 24]) + rng.choice([-1, 1]))
        elif emotion == "surprise":
            duration = seconds_per_beat * rng.choice([0.25, 0.5])
            velocity = int(rng.randint(70, 115))
            pitch = int(rng.choice(scale) + rng.choice([0, 12]))
        elif emotion == "fear":
            duration = seconds_per_beat * rng.choice([0.125, 0.25, 0.5])
            velocity = int(rng.randint(45, 75))
            pitch = int(rng.choice(scale) + rng.choice([-2, -1, 0, 1, 2]))
        elif emotion == "disgust":
            duration = seconds_per_beat * rng.choice([0.25, 0.75, 1.0])
            velocity = int(rng.randint(55, 95))
            pitch = int(rng.choice(scale) + rng.choice([0, 1, 2, -1]))
        elif emotion == "happy":
            duration = seconds_per_beat * rng.choice([0.25, 0.5, 1.0])
            velocity = int(rng.randint(75, 110))
            pitch = int(rng.choice(scale))
        elif emotion == "sad":
            duration = seconds_per_beat * rng.choice([0.5, 1.0, 1.5])
            velocity = int(rng.randint(50, 85))
            pitch = int(rng.choice(scale) - 12)
        else:
            duration = seconds_per_beat * rng.choice([0.5, 1.0])
            velocity = int(rng.randint(60, 100))
            pitch = int(rng.choice(scale))

        start = current_time
        end = min(current_time + duration, length_seconds)
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)
        current_time = end

    pm.instruments.append(instrument)
    pm.resolution = 960
    return pm


# ----------------------------
# Synthesis (MIDI → WAV, emotion-specific timbres)
# ----------------------------

def _linear_adsr(num_samples: int, sr: int, attack: float, release: float, sustain_level: float = 0.85) -> np.ndarray:
    total = num_samples
    a = int(max(1, min(total, attack * sr)))
    r = int(max(1, min(total - a, release * sr)))
    s = max(0, total - a - r)
    env = np.empty(total, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if s > 0:
        env[a : a + s] = sustain_level
    if r > 0:
        env[a + s :] = np.linspace(sustain_level, 0.0, r, endpoint=False)
    return env


def midi_to_wav(pm: pretty_midi.PrettyMIDI, sample_rate: int = 44100, emotion: str = "neutral") -> Tuple[int, np.ndarray]:
    duration = max([note.end for inst in pm.instruments for note in inst.notes], default=1.0)
    audio = np.zeros(int(sample_rate * duration) + 1, dtype=np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            freq = float(pretty_midi.note_number_to_hz(note.pitch))
            start_i = int(note.start * sample_rate)
            end_i = int(note.end * sample_rate)
            end_i = min(end_i, len(audio))
            n = max(1, end_i - start_i)
            tt = np.linspace(0.0, (n - 1) / sample_rate, n, endpoint=True)

            base_amp = 0.11 * (note.velocity / 127.0)

            if emotion == "happy":
                vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * tt)
                sig = (
                    0.75 * np.sin(2 * np.pi * (freq * vibrato) * tt)
                    + 0.25 * np.sin(2 * np.pi * (2 * freq) * tt)
                    + 0.15 * np.sin(2 * np.pi * (3 * freq) * tt)
                )
                env = _linear_adsr(n, sample_rate, 0.01, 0.12, 0.9)
            elif emotion == "sad":
                sig = 1.0 * np.sin(2 * np.pi * freq * tt)
                env = _linear_adsr(n, sample_rate, 0.02, 0.35, 0.7)
            elif emotion == "angry":
                partials = [1, 2, 3, 4, 5]
                amps = [1.0, 0.5, 0.3, 0.2, 0.15]
                sig = sum(a * np.sin(2 * np.pi * (freq * p) * tt) for p, a in zip(partials, amps))
                sig = np.tanh(1.2 * sig)  # mild distortion
                env = _linear_adsr(n, sample_rate, 0.005, 0.08, 0.85)
                base_amp *= 1.15
            elif emotion == "surprise":
                vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 7.0 * tt)
                sig = (
                    0.9 * np.sin(2 * np.pi * (freq * vibrato) * tt)
                    + 0.2 * np.sin(2 * np.pi * (2.5 * freq) * tt)
                )
                env = _linear_adsr(n, sample_rate, 0.005, 0.1, 0.85)
            elif emotion == "fear":
                tremolo = 0.75 + 0.25 * (0.5 * (1 + np.sin(2 * np.pi * 7.0 * tt)))
                sig = np.sin(2 * np.pi * freq * tt) * tremolo
                env = _linear_adsr(n, sample_rate, 0.02, 0.2, 0.8)
            elif emotion == "disgust":
                detune = 2 ** (3 / 1200.0)  # ~3 cents
                sig = 0.7 * np.sin(2 * np.pi * freq * tt) + 0.5 * np.sin(2 * np.pi * (freq * detune) * tt)
                gate = (np.sign(np.sin(2 * np.pi * 3.0 * tt)) + 1) / 2.0  # irregular gating
                sig *= 0.7 + 0.3 * gate
                env = _linear_adsr(n, sample_rate, 0.015, 0.15, 0.8)
            else:  # neutral
                sig = 0.9 * np.sin(2 * np.pi * freq * tt) + 0.2 * np.sin(2 * np.pi * 2 * freq * tt)
                env = _linear_adsr(n, sample_rate, 0.015, 0.15, 0.85)

            wave_note = base_amp * sig * env
            audio[start_i:end_i] += wave_note.astype(np.float32)

    max_val = float(np.max(np.abs(audio)))
    if max_val > 0:
        audio = (audio / max_val) * 0.92
    audio_int16 = np.int16(audio * 32767)
    return sample_rate, audio_int16


def play_wav_data(sample_rate: int, audio_int16: np.ndarray) -> Optional[pygame.mixer.Sound]:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
        sound = pygame.sndarray.make_sound(audio_int16)
        sound.play()
        return sound
    except Exception:
        return None


# ----------------------------
# UI helpers
# ----------------------------

def emotion_badge_html(emotion: str) -> str:
    color = EMOTION_COLORS.get(emotion, EMOTION_COLORS["neutral"])
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "surprise": "😮",
        "fear": "😨",
        "disgust": "🤢",
        "neutral": "😐",
    }
    emoji = emoji_map.get(emotion, "😐")
    return f"<span class='emotion-badge' style='background-color: {color}22; color: {color}; border: 1px solid {color}55;'>{emoji} {emotion.upper()}</span>"


# ----------------------------
# Session state
# ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "neutral"
if "sound" not in st.session_state:
    st.session_state.sound = None
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "use_deepface" not in st.session_state:
    st.session_state.use_deepface = HAS_DEEPFACE


# ----------------------------
# Sidebar
# ----------------------------
inject_base_css()

st.sidebar.markdown("### 🎛️ Options")
capture_mode = st.sidebar.radio("Capture mode", ["Webcam", "Image"], index=0)

if HAS_DEEPFACE:
    det_choice = st.sidebar.selectbox(
        "Emotion Detector",
        ["DeepFace (CNN)", "Heuristic (OpenCV)"]
    )
    st.session_state.use_deepface = det_choice.startswith("DeepFace")
else:
    st.sidebar.info("DeepFace not available. Using heuristic detector.")
    st.session_state.use_deepface = False

with st.sidebar.expander("Help", expanded=True):
    st.markdown(
        "- **Lighting**: Ensure good front lighting for best results.\n"
        "- **Emotions**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral.\n"
        "- **Playback**: If audio doesn't play, use the download button."
    )

with st.sidebar.expander("Emotions Covered"):
    cols = st.columns(2)
    for i, e in enumerate(EMOTIONS):
        with cols[i % 2]:
            st.markdown(f"- <span style='color:{EMOTION_COLORS[e]}; font-weight:700'>{e.title()}</span>", unsafe_allow_html=True)

with st.sidebar.expander("About / Version"):
    st.markdown(
        "**AI Music Composer** — Emotion-driven music generation.\n\n"
        f"- UI accent color: `{ACCENT}`\n"
        "- Version: 1.0 (2025)\n"
        "- Built with Streamlit, DeepFace, PrettyMIDI, pygame"
    )


# ----------------------------
# Title
# ----------------------------
st.markdown("<div class='app-title'>🎵 AI Music Composer from Face Emotions</div>", unsafe_allow_html=True)


# ----------------------------
# Main layout
# ----------------------------
set_dynamic_background(st.session_state.last_emotion)
left, right = st.columns([2, 1])

frame_placeholder = left.empty()
emotion_placeholder = right.empty()
audio_placeholder = right.empty()
controls_placeholder = right.empty()
download_placeholder = right.empty()


# ----------------------------
# Core workflows
# ----------------------------
def handle_emotion_change(new_emotion: str) -> None:
    st.session_state.last_emotion = new_emotion
    set_dynamic_background(new_emotion)
    pm = generate_midi_for_emotion(new_emotion, length_seconds=6.0)
    sr, audio_int16 = midi_to_wav(pm, sample_rate=44100, emotion=new_emotion)

    # Save to bytes for Streamlit audio and download
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    st.session_state.audio_bytes = bio.getvalue()

    # Browser audio widget
    with audio_placeholder:
        st.audio(st.session_state.audio_bytes, format="audio/wav")

    # Optional local playback via pygame
    try:
        if st.session_state.sound is not None:
            try:
                st.session_state.sound.stop()
            except Exception:
                pass
        st.session_state.sound = play_wav_data(sr, audio_int16)
    except Exception:
        pass

    with download_placeholder:
        st.download_button(
            label=f"Download {new_emotion.title()} Music",
            data=st.session_state.audio_bytes,
            file_name=f"{new_emotion}_music.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    # Small animation cue
    if new_emotion == "happy":
        st.balloons()


if capture_mode == "Webcam":
    st.sidebar.markdown("---")
    start = st.sidebar.button("▶️ Start Camera", use_container_width=True)
    stop = st.sidebar.button("⏹️ Stop Camera", use_container_width=True)

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
            st.session_state.running = False
        else:
            # Optional: smaller frame for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame from webcam")
                        break

                    # Detection
                    emotion = detect_emotion(frame, prefer_deepface=st.session_state.use_deepface)

                    # Draw face boxes for feedback
                    face_cascade = load_face_cascade()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
                    frame_display = frame.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")

                    # Emotion UI
                    badge_html = emotion_badge_html(emotion)
                    emotion_placeholder.markdown(
                        f"<div class='card'><div class='subtle'>😃 Detected emotion</div><div style='margin-top:6px'>{badge_html}</div></div>",
                        unsafe_allow_html=True,
                    )

                    if emotion != st.session_state.last_emotion:
                        handle_emotion_change(emotion)

                    # Playback controls
                    with controls_placeholder:
                        cols_ctrl = st.columns(2)
                        with cols_ctrl[0]:
                            if st.button("🔊 Play (pygame)", use_container_width=True):
                                if st.session_state.audio_bytes is not None:
                                    try:
                                        snd = pygame.mixer.Sound(buffer=st.session_state.audio_bytes)
                                        try:
                                            snd.play()
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                        with cols_ctrl[1]:
                            if st.button("⏹️ Stop", use_container_width=True):
                                try:
                                    pygame.mixer.stop()
                                except Exception:
                                    pass

                    time.sleep(0.1)
            finally:
                cap.release()
    else:
        left.info("Camera is stopped. Use the Start button in the sidebar to begin.")

else:  # Image mode
    uploaded = left.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    analyze = left.button("Analyze Emotion", use_container_width=True)

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img_rgb, channels="RGB")
        else:
            left.error("Failed to decode image.")

    if analyze and uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            left.error("Failed to decode image.")
        else:
            emotion = detect_emotion(img_bgr, prefer_deepface=st.session_state.use_deepface)
            badge_html = emotion_badge_html(emotion)
            emotion_placeholder.markdown(
                f"<div class='card'><div class='subtle'>😃 Detected emotion</div><div style='margin-top:6px'>{badge_html}</div></div>",
                unsafe_allow_html=True,
            )
            handle_emotion_change(emotion)


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Your Name — 2025")
