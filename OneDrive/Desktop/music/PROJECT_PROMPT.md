# 🎭 AI Music Composer from Face Emotions - Complete Project Prompt

## 📋 Project Overview
Build a real-time Streamlit web application that captures user facial expressions via webcam, detects emotions using computer vision, and automatically generates unique music based on the detected emotion. The app synthesizes music on-the-fly and allows users to download generated compositions.

---

## 🎯 Goals & Objectives

### Primary Goals
1. **Real-time Emotion Detection**: Detect 7 core human emotions from facial expressions without requiring external API calls or large model downloads
2. **Music Generation**: Create unique, emotion-specific MIDI melodies mapped to each emotion
3. **Audio Synthesis**: Convert generated MIDI to playable WAV format with real-time audio synthesis
4. **User-Friendly Interface**: Build an intuitive Streamlit UI with live webcam feed, emotion display, auto-play music, and download functionality
5. **Modular Architecture**: Keep code organized with separate functions for emotion detection, music generation, audio synthesis, and playback

### Success Criteria
- ✅ App runs with single command: `streamlit run app.py`
- ✅ All 7 emotions detected and mapped to unique music
- ✅ Music generates and plays within 2 seconds of emotion change
- ✅ Users can download generated WAV files
- ✅ No external API calls or large model downloads required
- ✅ Works on Windows/Mac/Linux with Python 3.8+

---

## 🎭 The 7 Emotions & Their Characteristics

### 1. **Happy** 😊
- **Facial Markers**: Smile (bright mouth), open eyes, good face contrast
- **Music**: Fast tempo (140 BPM), C major scale, bright and uplifting
- **Scale**: [60, 62, 64, 65, 67, 69, 71, 72] (C major)
- **Duration**: Variable note lengths (0.25, 0.5, 1.0 beats)
- **Velocity**: 70-110 (bright, energetic)

### 2. **Sad** 😢
- **Facial Markers**: Frown, low contrast, closed/droopy eyes
- **Music**: Slow tempo (60 BPM), minor scale, melancholic
- **Scale**: [60, 62, 63, 65, 67, 68, 70, 72] (C minor)
- **Duration**: Longer note lengths (0.5, 1.0, 1.5 beats)
- **Velocity**: 50-80 (soft, subdued)

### 3. **Angry** 😠
- **Facial Markers**: Low brightness, high contrast, tense expression, edge density
- **Music**: Medium-fast tempo (120 BPM), dissonant intervals, intense
- **Scale**: [60, 61, 63, 64, 66, 67, 69, 70] (dissonant chromatic)
- **Duration**: Short, punchy notes (0.25, 0.5, 1.0 beats)
- **Velocity**: 80-127 (loud, aggressive)

### 4. **Surprise** 😮
- **Facial Markers**: Very wide eyes (high eye variance), mouth open, moderate brightness
- **Music**: Very fast tempo (160 BPM), large jumps between notes, playful
- **Scale**: [60, 64, 67, 72, 76, 79] (sparse, jumping intervals)
- **Duration**: Short notes (0.25, 0.5 beats)
- **Velocity**: 60-110 (varied, energetic)

### 5. **Fear** 😨
- **Facial Markers**: Very wide eyes (eye variance > 350), mouth open, tense (high edge density)
- **Music**: Moderate tempo (100 BPM), trembling effect, uncertain rhythm, sparse notes
- **Scale**: [60, 63, 67, 70] (minor triad, sparse)
- **Duration**: Very short, trembling notes (0.125, 0.25, 0.5 beats)
- **Velocity**: 40-70 (soft, nervous)

### 6. **Disgust** 🤢
- **Facial Markers**: Wrinkled nose (high nose variance), low brightness, high face contrast
- **Music**: Slower tempo (80 BPM), chromatic off-notes, irregular rhythm, unpleasant
- **Scale**: [60, 61, 63, 65, 68] (chromatic, off-scale)
- **Duration**: Irregular (0.25, 0.75, 1.0 beats)
- **Velocity**: 50-90 (moderate, uncomfortable)

### 7. **Neutral** 😐
- **Facial Markers**: Balanced features, no strong expression, relaxed
- **Music**: Medium tempo (100 BPM), balanced C major scale, calm
- **Scale**: [60, 62, 64, 65, 67, 69, 71] (C major, balanced)
- **Duration**: Regular notes (0.5, 1.0 beats)
- **Velocity**: 55-95 (balanced)

---

## 🏗️ Architecture & Modules

### 1. **Emotion Detection Module** (`detect_emotion()`)
**Input**: BGR frame from OpenCV  
**Output**: Emotion string (happy, sad, angry, neutral, surprise, fear, disgust)

**Method**: Facial feature analysis using OpenCV Haar Cascade + image processing
- Detects faces in grayscale image
- Extracts regions: eyes (upper third), mouth (lower half), nose (middle)
- Calculates metrics:
  - **Eye variance**: Openness of eyes
  - **Mouth intensity**: Brightness of mouth region (smile indicator)
  - **Nose variance**: Wrinkles (disgust)
  - **Face contrast**: Overall expression intensity
  - **Brightness**: Overall lighting
  - **Edge density**: Tension in expression
- Applies decision tree logic to classify emotion

**Key Advantages**:
- No external model downloads
- Real-time processing (~30ms per frame)
- Works with any webcam
- Built-in OpenCV cascades (pre-trained)

### 2. **Music Generation Module** (`generate_midi_for_emotion()`)
**Input**: Emotion string, optional duration (seconds), optional tempo  
**Output**: PrettyMIDI object with note sequence

**Process**:
1. Select emotion-specific scale and BPM
2. Generate random note sequence with:
   - Random note duration from emotion-specific set
   - Random velocity from emotion-specific range
   - Random pitch from emotion-specific scale
   - Emotion-specific variations (e.g., fear trembling, disgust chromatic)
3. Build MIDI with single piano instrument (program 0)
4. Set resolution to 960 ticks per quarter note (standard MIDI)

**Emotion-Music Mapping**:
```
Happy    → Major scale, fast, bright velocity
Sad      → Minor scale, slow, soft velocity
Angry    → Dissonant, medium-fast, loud velocity
Surprise → Sparse jumpy scale, very fast, varied velocity
Fear     → Minor triad, medium, very short notes, low velocity
Disgust  → Chromatic, slow, irregular, uncomfortable velocity
Neutral  → Major scale, medium, balanced velocity
```

### 3. **Audio Synthesis Module** (`midi_to_wav()`)
**Input**: PrettyMIDI object  
**Output**: Tuple (sample_rate, audio_int16 numpy array)

**Process**:
1. Calculate total duration from MIDI notes
2. Create time array at 44.1 kHz sample rate
3. For each note in MIDI:
   - Convert MIDI pitch number to frequency (Hz)
   - Generate sine wave for note duration
   - Apply envelope (fade-in/fade-out) to prevent clicks
   - Scale by velocity (0-127 → 0-1 amplitude)
   - Mix into output audio buffer
4. Normalize audio to -1.0 to 1.0 range
5. Convert to 16-bit PCM integer format

**Key Details**:
- Simple additive synthesis (sine waves only)
- Linear fade envelope (eliminates audio artifacts)
- Normalization prevents clipping
- 44.1 kHz (CD quality) sample rate

### 4. **Audio Playback Module** (`play_wav_data()`)
**Input**: Sample rate, audio_int16 numpy array  
**Output**: pygame.mixer.Sound object

**Process**:
1. Initialize pygame mixer at correct sample rate
2. Create pygame Sound object from audio array
3. Call play() to start playback
4. Return sound object for later stop/control

### 5. **Streamlit UI Module**
**Components**:
- **Page Title**: "AI Music Composer from Face Emotions"
- **Sidebar Controls**:
  - "Start Camera" button → Initiate webcam capture loop
  - "Stop Camera" button → Exit capture loop
- **Main Area**:
  - **Frame placeholder**: Live webcam feed with face detection boxes (green rectangles)
  - **Emotion placeholder**: Real-time emotion display with emoji
  - **Audio placeholder**: Audio player for generated WAV
  - **Download button**: Save current emotion's music as WAV file

**Session State Management**:
- `st.session_state.running`: Boolean for webcam loop control
- `st.session_state.last_emotion`: Track previous emotion for change detection
- `st.session_state.sound`: Current playing pygame sound object (for stopping previous)

**Main Loop** (when running):
1. Capture frame from webcam (`cv2.VideoCapture(0)`)
2. Detect emotion from frame
3. Draw face detection boxes on frame for feedback
4. Display frame
5. If emotion changed:
   - Generate MIDI for new emotion
   - Convert MIDI to WAV
   - Write WAV to temporary file
   - Play audio via pygame
   - Update Streamlit audio player
   - Show download button
6. Sleep 100ms to prevent busy loop
7. Release camera on stop

---

## 📦 Dependencies & Requirements

### Python Packages
```
streamlit           # Web app framework
opencv-python       # Computer vision, face detection
pretty_midi          # MIDI generation and manipulation
pygame              # Audio playback
numpy               # Numerical computing
tf-keras            # Keras compatibility (for DeepFace if needed)
```

### System Requirements
- **Python**: 3.8 or later
- **OS**: Windows, macOS, or Linux
- **RAM**: 2 GB minimum (3+ GB recommended)
- **Camera**: USB webcam or built-in camera
- **Audio**: Speaker or headphone output

### Model/Data Files
- **None required!** All detection uses OpenCV built-in Haar Cascades (pre-installed with opencv-python)

---

## 🚀 Installation & Setup

### Step 1: Create Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```

### Step 4: Access in Browser
- Open: `http://localhost:8501`
- If not automatic, Streamlit will print the URL in terminal

### Step 5: Grant Camera Permission (if prompted)
- Windows: Allow app in Settings → Privacy → Camera
- macOS: Grant camera access when prompted
- Linux: Usually automatic

---

## 🎮 Usage Guide

### Basic Workflow
1. **Open App**: Navigate to http://localhost:8501
2. **Click "Start Camera"**: Begin capturing face
3. **Allow Camera Access**: Grant permission if prompted
4. **See Live Feed**: Webcam video appears with green face detection boxes
5. **Express Emotion**: Make facial expressions (smile, frown, open mouth, etc.)
6. **Watch Emotion Detect**: Emotion text updates in real-time
7. **Hear Music**: Music auto-generates and plays within 1-2 seconds
8. **Download**: Click "Download Music" button to save WAV file
9. **Try All 7 Emotions**: Repeat steps 5-8 for each emotion

### Tips for Best Detection
- **Good Lighting**: Face should be well-lit (avoid shadows)
- **Clear Face**: Position face clearly in center of frame
- **Exaggerate Expressions**: More pronounced emotions = better detection
- **Wait 1-2 Seconds**: Emotion detection needs time to stabilize
- **Different Camera Index**: If webcam not detected, change `cv2.VideoCapture(0)` to `1` or `2`

---

## 🎵 Generated Music Examples

### Happy Music
- Tempo: 140 BPM
- Scale: C major [60, 62, 64, 65, 67, 69, 71, 72]
- Effect: Uplifting, bright, energetic
- Typical sound: Cheerful piano melody

### Sad Music
- Tempo: 60 BPM
- Scale: C minor [60, 62, 63, 65, 67, 68, 70, 72]
- Effect: Melancholic, slow, gentle
- Typical sound: Soft, sorrowful piano

### Angry Music
- Tempo: 120 BPM
- Scale: Dissonant [60, 61, 63, 64, 66, 67, 69, 70]
- Effect: Intense, loud, aggressive
- Typical sound: Forceful, dissonant notes

### Surprise Music
- Tempo: 160 BPM
- Scale: Sparse jumps [60, 64, 67, 72, 76, 79]
- Effect: Playful, unexpected, energetic
- Typical sound: Quick, jumping piano phrases

### Fear Music
- Tempo: 100 BPM
- Scale: Minor triad [60, 63, 67, 70]
- Effect: Trembling, uncertain, sparse
- Typical sound: Short, shaky, uncertain notes

### Disgust Music
- Tempo: 80 BPM
- Scale: Chromatic off-notes [60, 61, 63, 65, 68]
- Effect: Unpleasant, irregular, uncomfortable
- Typical sound: Awkward, discordant piano

### Neutral Music
- Tempo: 100 BPM
- Scale: C major [60, 62, 64, 65, 67, 69, 71]
- Effect: Calm, balanced, centered
- Typical sound: Pleasant, balanced piano melody

---

## 🛠️ Customization & Improvements

### Easy Customizations
1. **Change Music Duration**: Modify `length_seconds=6` in `generate_midi_for_emotion()`
2. **Adjust Emotion Sensitivity**: Tune threshold values in `detect_emotion()` (brightness, variance, etc.)
3. **Change Instrument**: Modify MIDI program number in `generate_midi_for_emotion()` (0=Piano, 33=Acoustic Bass, etc.)
4. **Adjust Audio Volume**: Scale velocity ranges in music generation
5. **Change Camera Index**: If webcam not detected, change `cv2.VideoCapture(0)` to `1`, `2`, etc.

### Advanced Improvements
1. **Replace Sine Synthesis with FluidSynth**: Use soundfont files for richer audio
2. **Add Multiple Instruments**: Layer instruments (bass, drums, melody)
3. **Improve Face Detection**: Use DNN-based face detector instead of Haar Cascade
4. **Add ML-based Emotion Detection**: Integrate pre-trained CNN (DeepFace, FER2013, etc.)
5. **Save Emotion History**: Log detected emotions over time with timestamps
6. **Add Smooth Transitions**: Crossfade between emotion changes instead of abrupt switches
7. **Add Visual Effects**: Draw emotion-specific overlays or filters
8. **Add MIDI Download**: Allow downloading raw MIDI files, not just WAV
9. **Add Real-time Recording**: Record all generated music to a session file
10. **Multi-face Support**: Detect multiple faces and generate ensemble music

---

## 📁 Project File Structure

```
music/
├── app.py                      # Main Streamlit application (single file)
├── requirements.txt            # Python dependencies
├── README.md                   # User-facing documentation
├── PROJECT_PROMPT.md          # This file (complete project specification)
├── run_streamlit_helper.py    # Helper script to start Streamlit non-interactively
├── streamlit_helper.pid       # PID file for tracking running process
├── .venv/                     # Virtual environment (optional)
└── __pycache__/               # Python bytecode cache
```

---

## 🔧 Troubleshooting

### Issue: "Could not open webcam"
**Solution**: 
- Check camera permissions in OS settings
- Try changing camera index: `cv2.VideoCapture(1)` or `2`
- Ensure no other app is using the camera

### Issue: No emotion detected (shows "neutral" always)
**Solution**:
- Ensure good lighting
- Exaggerate facial expressions
- Tune threshold values in `detect_emotion()` function

### Issue: Audio playback fails (Pygame mixer error)
**Solution**:
- Install pygame audio drivers: `pip install --upgrade pygame`
- Use download button to save WAV and play externally
- Check system audio settings

### Issue: App runs slowly (lag in video)
**Solution**:
- Reduce video frame size: Add `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)`
- Reduce emotion detection frequency: Increase `time.sleep()` value
- Close other applications

### Issue: "tensorflow" or "keras" errors (if using DeepFace)
**Solution**:
- Install compatibility package: `pip install tf-keras`
- Current app uses OpenCV, so this shouldn't occur

### Issue: Streamlit asks for email on startup
**Solution**:
- Press Enter to skip
- Or run helper script: `python run_streamlit_helper.py`

---

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| Emotion Detection Speed | ~30 ms per frame |
| Music Generation Time | ~100 ms |
| MIDI-to-WAV Synthesis | ~200-500 ms (depending on duration) |
| Total Emotion→Music Latency | ~1-2 seconds |
| Audio Playback Latency | <100 ms |
| CPU Usage (avg) | 15-25% (single core) |
| RAM Usage (avg) | 200-300 MB |
| Supported Framerate | 30 FPS (OpenCV default) |

---

## 🎓 Learning Outcomes

This project teaches:
1. **Computer Vision**: Face detection, image processing (OpenCV)
2. **Music Generation**: MIDI creation, music theory basics (pretty_midi)
3. **Audio Processing**: Waveform synthesis, audio format conversion (NumPy)
4. **Web Development**: Interactive UI with Streamlit
5. **State Management**: Session state in web apps
6. **Real-time Processing**: Continuous data pipeline (webcam → detection → generation → playback)
7. **Python Best Practices**: Modular functions, error handling, resource management

---

## 📚 References & Resources

### Libraries & Documentation
- Streamlit: https://streamlit.io/docs
- OpenCV: https://opencv.org/
- PrettyMIDI: https://github.com/craffel/pretty-midi
- Pygame: https://www.pygame.org/
- NumPy: https://numpy.org/

### Music Theory
- MIDI Note Numbers: C4 (Middle C) = 60
- Common Scales:
  - Major: [0, 2, 4, 5, 7, 9, 11] intervals
  - Minor: [0, 2, 3, 5, 7, 8, 10] intervals
  - Pentatonic: [0, 2, 4, 7, 9] intervals

### Emotion Recognition Research
- Ekman's 7 Basic Emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- Facial Action Coding System (FACS) by Ekman & Friesen
- FER2013 Dataset: Public facial expression recognition dataset

### Audio Processing
- Sample Rate: 44.1 kHz (CD quality)
- Bit Depth: 16-bit (PCM)
- Envelope: Attack-Decay-Sustain-Release (ADSR) model

---

## 📝 Changelog & Version History

### v1.0 (October 22, 2025)
- Initial release
- 7 emotions supported: happy, sad, angry, neutral, surprise, fear, disgust
- Real-time face detection with Haar Cascade
- Emotion-based MIDI generation with unique musical characteristics
- Real-time audio synthesis (sine wave additive synthesis)
- Streamlit UI with live webcam feed and emotion display
- Audio playback via Pygame
- Download WAV files
- No external API calls or large model downloads required

---

## 📞 Support & Contributing

### Reporting Issues
Include:
- OS (Windows/Mac/Linux)
- Python version
- Exact error message
- Steps to reproduce

### Feature Requests
Suggest improvements in project documentation or GitHub issues

### Contributing
Welcome to:
- Improve emotion detection thresholds
- Add more instruments/sound variations
- Optimize performance
- Add tests
- Improve documentation

---

## 📄 License

This project is provided as-is for educational and personal use.

---

## 🎉 Conclusion

This AI Music Composer demonstrates the power of combining computer vision, music generation, and audio synthesis in a real-time, user-friendly application. It showcases practical machine learning concepts without requiring complex model training or large dataset downloads.

**Happy composing! 🎵**

---

*Last Updated: October 22, 2025*  
*Project Status: Complete and Running*
