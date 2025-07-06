# 🎥 ClipToReel

**ClipToReel** is an AI-powered video intelligence system that automatically generates short, high-impact highlight reels from long-form podcasts or videos based on virality and semantic relevance. It combines modern video/audio processing, deep learning-based virality prediction, and large language models (LLMs) to evaluate and extract the most engaging moments in a video.

---

## 🌐 Live Demo (Optional)

Add a demo GIF or screen recording showing the workflow from uploading a video to downloading viral clips.

---

## 🚀 Features

* 🎮 Smart Highlight Detection using audio patterns and duration
* 🤖 LLM-based relevance scoring for contextual importance
* 📊 Virality Prediction Model trained on metadata features
* 📁 Multi-format input support (MP4, MP3, WAV)
* 📊 Interactive Frontend built with React + Tailwind CSS
* 💾 Downloadable short clips for social media or reels
* 🔄 Modular architecture (frontend/backend separation)

---

## 🎓 Architecture Overview

```
            [Frontend - ReactJS Upload UI]       
                        |
        +---------------+----------------+
        | Upload Podcast / Video / Audio |
        +---------------+----------------+
                        |
              POST /api/upload
                        |
         [Backend - Flask API Server]
                        |
    +-------------------|------------------------+
    |   Video Input Handler                      |
    |   Smart Highlight Selector (scoring.py)    |
    |   LLM Relevance Ranker                     |
    |   Virality Classifier (XGBoost model)      |
    +-------------------|------------------------+
                        |
               Returns Ranked + Labeled Clips
                        |
               [Frontend Preview UI]
```

---

## 📊 Tech Stack

### Backend

* Python 3.10+
* Flask + Flask-CORS
* OpenAI Whisper (for transcription)
* spaCy / TextBlob / Sentence Transformers
* XGBoost, Scikit-learn
* MoviePy / FFmpeg / Pydub
* LLM (via `ollama`, or others)

### Frontend

* ReactJS (with Vite or CRA)
* Tailwind CSS
* Axios

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Rajat-Projects/ClipToReel.git
cd ClipToReel
```

### 2. Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Backend Server

```bash
python main.py
```

Runs the API at `http://127.0.0.1:5000/`

### 4. Setup Frontend

```bash
cd ../frontend
npm install
npm run dev
```

---

## 👁️ Usage Guide

### Upload and Process

* Select a `.mp3` or `.mp4` file via the React UI
* It triggers `/api/upload` followed by `/api/process`
* Shows table of clips with:

  * LLM Score
  * Virality Score
  * Viral/Not Viral label
  * Download and View options

### How It Works

1. **Transcribe Audio**: Whisper or pydub extracts transcript.
2. **Segment**: Audio/video segmented into short clips.
3. **Score**: Each clip is scored using:

   * Relevance (via LLM)
   * Virality model (`virality_model.pkl`)
4. **Select Highlights**: Top clips are chosen and returned.

---

## 🌊 Virality Model Training

### File: `train_virality_model.py`

* Extracts features like energy, duration, semantic embedding distance
* Trains an XGBoost classifier
* Saves model as `virality_model.pkl`

To retrain:

```bash
python train_virality_model.py
```

---

## 📂 Project Structure

```
ClipToReel/
├── backend/
│   ├── main.py                        # Entry point
│   ├── video_input_handler.py
│   ├── smart_highlight_selector.py
│   ├── llm_ranker_improved.py
│   ├── scoring.py                     # Custom score logic
│   ├── auto_label_unlabeled_clips.py
│   ├── train_virality_model.py       # Model training script
│   ├── virality_model.pkl            # Saved classifier
├── frontend/
│   ├── components/
│   │   └── UploadForm.jsx            # React upload & results UI
│   ├── index.css                     # Tailwind styling
│   └── App.jsx
├── input/, audio_clips/, waveforms/  # Input & generated media
├── requirements.txt
```

---

## 👤 Author

**Rajat Pednekar**
[GitHub](https://github.com/Rajat-Projects)

---

## 🌐 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🙋 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---

## ✨ Future Enhancements

* Add support for YouTube URL input
* Integrate TTS for highlight voiceover
* Auto-publish to Instagram Reels / TikTok
* UI to edit and stitch clips manually
