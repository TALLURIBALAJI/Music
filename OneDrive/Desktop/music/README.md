# AI Music Composer from Face Emotions

This Streamlit app captures webcam frames, detects the user's dominant emotion using DeepFace, and generates simple music based on the detected emotion. The generated music is synthesized to WAV and can be played and downloaded.

## Features

- Live webcam feed (OpenCV)
- Emotion detection (DeepFace)
- Emotion -> music mapping (PrettyMIDI for note logic, simple synthesis to WAV)
- Auto-play generated music and download

## Requirements

Install with:

```powershell
pip install -r requirements.txt
```

Note: DeepFace will download pre-trained models on first run; ensure you have an internet connection.

## Run

```powershell
streamlit run app.py
```

## Notes and caveats

- The app synthesizes audio using a simple additive sine approach to avoid requiring FluidSynth.
- Pygame is used to play the generated WAV audio. On some systems you may need to install additional audio drivers.
- If you encounter permission or webcam access issues on Windows, ensure apps have camera permission.

## Files

- `app.py`: Main Streamlit application (single-file deliverable)
- `requirements.txt`: Python package list

## Next steps / improvements

- Use a proper synthesizer (FluidSynth) and soundfonts for richer audio
- Add smoother transitions between emotions
- Improve UI and add presets

Enjoy!
