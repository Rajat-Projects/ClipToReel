from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import uuid
import os
import joblib

# Import core pipeline modules
from app.video_input_handler import extract_audio_from_video, cut_highlight_video_clips
from app.smart_highlight_selector import smart_highlight_selector
from app.llm_ranker import llm_score_local
from app.scoring import compute_virality_score
from app import logger

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Define base directory and resolve paths reliably
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Load model once
model = joblib.load(BASE_DIR / "app/model/virality_model.pkl")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running!"})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # 🧹 Step 1: Clean uploads directory
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            f.unlink()

    # 🧹 Step 2: Clean transcripts directory
    transcript_dir = Path("transcripts")
    if transcript_dir.exists():
        for f in transcript_dir.glob("*.json"):
            f.unlink()

    # 💾 Step 3: Save the new uploaded file
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    file.save(str(filepath))

    return jsonify({"message": "File uploaded", "filename": filename})


@app.route("/api/process", methods=["POST"])
def process_file():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    video_path = UPLOAD_DIR / filename
    audio_path = video_path.with_suffix(".wav")

    extract_audio_from_video(video_path, audio_path)
    highlights = smart_highlight_selector(audio_path)

    cut_highlight_video_clips(video_path, highlights, model, STATIC_DIR)

    response_data = []

    for i, h in enumerate(highlights):
        features = [[
            h["llm_score"],
            h["sentiment"],
            int(h["keyword_hit"]),
            h["length_sec"]
        ]]
        predicted_label = model.predict(features)[0]

    # Append the structured response with clip path and label
        clip_data = {
            "text": h["text"],
            "score": h["virality_score"],
            "start": h["start"],
            "end": h["end"],
            "llm_score": h["llm_score"],
            "sentiment": h["sentiment"],
            "keyword_hit": h["keyword_hit"],
            "length_sec": h["length_sec"],
            "label": int(predicted_label),
            "clip_path": f"static/clip_{i+1}.mp4"  # Or the correct path from your cutter
        }
        response_data.append(clip_data)

    return jsonify({"highlights": response_data})


@app.route("/static/<path:filename>")
def serve_static_file(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
