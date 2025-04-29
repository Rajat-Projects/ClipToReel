# 🎹 Podcast-to-Reels Generator (Local, Open-Source)

## 📋 Project Overview

**Podcast-to-Reels** is a fully **local**, **open-source** Python app that:
- Transcribes long podcast episodes (.mp3 or .mp4) using OpenAI’s Whisper.
- Detects highlight moments using NLP techniques (spaCy).
- Cuts those highlights into **short audio/video clips** automatically.
- For audio-only podcasts, it **creates dynamic waveform videos** as reels.
- Adds optional subtitles to make reels social-media-ready.

Built for **YouTube Shorts**, **Instagram Reels**, **TikTok**, and more!

---

## 🛠 Folder Structure

```
podcast-to-reels/
│
├── input/             → 🎷 Drop full podcast files (.mp3/.mp4)
├── output/            → 🎥 Generated short reels (.mp4)
├── transcripts/       → 📝 Saved transcription files (.json)
├── audio_clips/       → 🎵 Temporary extracted audio segments
├── waveforms/         → 🎨 Temporary waveform visualizations
├── podcastenv/        → Virtual environment (Python venv)
├── requirements.txt   → Required Python libraries
└── main.py            → Master script (transcription → highlight → reel generation)
```

---

## 🔧 Setup Instructions

1. Clone or create the `podcast-to-reels/` folder manually.
2. Open terminal inside the project folder.
3. Create and activate a virtual environment:

```bash
python -m venv podcastenv
source podcastenv/bin/activate      # (Mac/Linux)
podcastenv\Scripts\activate         # (Windows)
```

4. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

5. Add your `.mp3` (or `.mp4`) podcast files to the `input/` folder.
6. Run the app:

```bash
python main.py
```

---

## ✨ Key Features

- 🎷 **Whisper Transcription** — fast and accurate.
- 🧐 **NLP Highlight Detection** — finds engaging moments automatically.
- ✂️ **Clip Extraction** — cuts podcast into mini reels.
- 🎨 **Dynamic Waveform Video Creation** — makes reels lively even for audio-only podcasts.
- 🔤 **Optional Subtitle Overlay** — adds captions for better engagement.
- ⚡ **Fast, Local, Free** — no API costs, no data leaks.

---

## 📈 Whisper Model Options

When running the app, you can select different Whisper models depending on your machine:

| Model  | Speed (Fast → Slow) | Accuracy (Low → High) | Best for |
|--------|---------------------|----------------------|----------|
| tiny   | 🟢🟢🟢🟢 | 🔴🔴 | Very fast tests |
| small  | 🟢🟢🟢 | 🔴🔴🔴 | Real-world podcasts |
| base   | 🟢🟢 | 🔴🔴🔴🔴 | Balanced |
| medium | 🟢 | 🔴🔴🔴🔴🔴 | High-quality work |
| large  | 🔴 | 🔴🔴🔴🔴🔴 | Research-grade, GPU needed |

---

## ⚠️ Known Limitations

- Very large podcast files (90+ minutes) may take longer to process.
- Whisper model loading may consume a lot of RAM (suggest using `small` or `tiny` models for slower machines).
- Some subtitle timing inaccuracies for very noisy podcasts.
- No direct YouTube download support (coming in future upgrades).

---

## 🚀 Future Improvements (Planned)

- GUI interface (Streamlit/Tkinter)
- Auto upload clips to Instagram/TikTok/YouTube
- Multiple design templates for reels
- Sentiment-based highlight detection (more emotional reels)

---

## 🧑‍💻 Author

Built with ❤️ by [Your Name Here]

---

# 📢 Quick Start Summary

```bash
# Terminal commands:
cd podcast-to-reels
python -m venv podcastenv
source podcastenv/bin/activate      # (Mac/Linux) OR podcastenv\Scripts\activate (Windows)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Drop your .mp3 into input/
python main.py
```

👉 Your Reels are ready in the `output/` folder! 🚀

