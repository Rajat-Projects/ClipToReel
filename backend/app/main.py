import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import whisper
import json
import spacy
from textblob import TextBlob
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import AudioFileClip, ImageSequenceClip
from .smart_highlight_selector import smart_highlight_selector
from pydub.silence import detect_nonsilent
from .video_input_handler import extract_audio_from_video, cut_highlight_video_clips
from .logger import log_clip_data, log_viral_clip_only
import joblib
from pathlib import Path

base_dir = Path(__file__).resolve().parent  # backend/app/
model_path = base_dir / "model" / "virality_model.pkl"
virality_model = joblib.load(model_path)


# === 1. Utility: Create folders if missing ===
def setup_folders():
    folders = ["input", "output", "transcripts", "audio_clips", "waveforms"]
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)

# === 2. Model Selection ===
def select_whisper_model():
    print("\n🔵 Whisper Model Selection:")
    print("1. tiny    (fastest, less accurate)")
    print("2. small   (fast, decent accuracy)")
    print("3. base    (default balanced)")
    print("4. medium  (better accuracy)")
    print("5. large   (best accuracy, needs big GPU)")

    choice = input("Select model (1-5): ").strip()
    model_map = {
        "1": "tiny",
        "2": "small",
        "3": "base",
        "4": "medium",
        "5": "large"
    }
    model_name = model_map.get(choice, "base")
    print(f"\n✅ You selected '{model_name}' model.\n")
    return model_name

# === 3. Transcribe a single podcast file ===
def transcribe_audio(file_path, model):
    print(f"🎧 Transcribing: {file_path.name} ...")

    audio_duration = get_audio_duration(file_path)

    if audio_duration > 3600:  # 3600 seconds = 1 hour
        print("⚠️ Warning: Large file detected (>60 minutes). This may take some time...\n")

    result = model.transcribe(str(file_path))

    # Save transcript
    transcript_path = Path("transcripts") / f"{file_path.stem}.json"
    with open(transcript_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Saved transcription: {transcript_path}\n")
    return result

# === 4. Helper: Get audio duration ===
def get_audio_duration(file_path):
    import ffmpeg
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        print(f"❗ Error getting audio duration: {e}")
        return 0

# === 5. Main pipeline ===
def main():
    print("🎬 Podcast-to-Reels: Transcription Phase Started!\n")
    setup_folders()

    input_files = list(Path("input").glob("*.mp3")) + list(Path("input").glob("*.mp4")) + list(Path("input").glob("*.mov"))
    if not input_files:
        print("❗ No media files found in 'input/' folder.")
        return

    model_name = select_whisper_model()
    print("⏳ Loading Whisper model...")
    model = whisper.load_model(model_name)
    print("✅ Model loaded successfully.\n")

    for file_path in input_files:
        try:
            if file_path.suffix in ['.mp4', '.mov']:
                extracted_audio = extract_audio_from_video(file_path, Path("input/temp_audio.wav"))
                audio_for_pipeline = extracted_audio
            else:
                audio_for_pipeline = file_path

            transcribe_audio(audio_for_pipeline, model)
            transcript_path = Path("transcripts") / f"{audio_for_pipeline.stem}.json"
            highlights = smart_highlight_selector(transcript_path, min_clip_duration=20, max_clip_duration=90)

            if file_path.suffix in ['.mp4', '.mov']:
                cut_highlight_video_clips(file_path, highlights, virality_model=virality_model)
            else:
                cut_audio_clips(audio_for_pipeline, highlights)
                audio_clip_files = list(Path("audio_clips").glob(f"{audio_for_pipeline.stem}_clip_*.mp3"))
                for clip_file in audio_clip_files:
                    generate_waveform_video(clip_file)

        except Exception as e:
            print(f"❗ Error processing {file_path.name}: {e}")

    print("\n🎉 All podcasts processed! Transcripts and clips saved.")


# === Smart Highlight Detection ===
def detect_highlights(transcript_json_path):
    print(f"🧠 Detecting smart highlights in {transcript_json_path.name} ...")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    important_keywords = ["success", "growth", "failure", "habit", "mindset", "motivation", "leadership", "achievement"]

    with open(transcript_json_path, "r") as f:
        transcript = json.load(f)

    highlights = []

    for segment in transcript["segments"]:
        text = segment["text"]
        doc = nlp(text)

        # Sentiment scoring
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # Range: -1 (negative) to +1 (positive)

        # Keyword matching
        keyword_match = any(keyword in text.lower() for keyword in important_keywords)

        # Length check
        long_enough = len(text.split()) > 10

        # Named entities check
        has_entities = len(doc.ents) >= 2

        # Decision logic
        if sentiment >= 0.5 or sentiment <= -0.5 or keyword_match or long_enough or has_entities:
            highlights.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "sentiment": sentiment,
                "keyword_match": keyword_match
            })

    print(f"✅ Found {len(highlights)} smart highlight segments.\n")
    return highlights



# === Cut audio clips from highlights ===
def cut_audio_clips(audio_file_path, highlights):
    print(f"\n✂️ Cutting clips from {audio_file_path.name} ...")
    audio = AudioSegment.from_file(audio_file_path)

    buffer_time = 1500
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    for idx, h in enumerate(highlights):
        start = int(h["start"] * 1000)
        end = int(h["end"] * 1000)

        new_end = end + buffer_time
        for (ns_start, ns_end) in nonsilent_ranges:
            if ns_start >= end:
                new_end = min(ns_start + buffer_time, len(audio))
                break
        new_end = min(new_end, len(audio))

        clip = audio[start:new_end]
        out_path = Path("audio_clips") / f"{audio_file_path.stem}_clip_{idx+1}.mp3"
        clip.export(out_path, format="mp3")
        print(f"✅ Saved clip: {out_path.name} | Duration: {round((new_end-start)/1000, 1)}s")

        # Prepare features & predict
        features = [[
            h["llm_score"],
            h["sentiment"],
            h["length_sec"],
            int(h["keyword_hit"])
        ]]
        label = int(virality_model.predict(features)[0])

        if h["text"] in existing_texts:
            print(f"⏩ Skipping duplicate clip: {h['text'][:30]}...")
            continue

        clip_data = {
            "llm_score": h["llm_score"],
            "sentiment": h["sentiment"],
            "length_sec": h["length_sec"],
            "keyword_hit": int(h["keyword_hit"]),
            "text": h["text"],
            "clip_path": str(out_path),
            "predicted_score": h["virality_score"],
            "label": label
        }

        log_clip_data(clip_data)

        if label == 1:
            log_viral_clip_only(clip_data)

# === Generate Dynamic Waveform Video ===
def generate_waveform_video(audio_clip_path):
    print(f"🎨 Generating waveform video for {audio_clip_path.name} ...")

    # Load audio
    audio = AudioSegment.from_file(audio_clip_path)
    samples = np.array(audio.get_array_of_samples())

    # Normalize samples
    samples = samples / np.max(np.abs(samples))

    frame_rate = 25  # 25 fps
    clip_duration = len(samples) / audio.frame_rate
    total_frames = int(clip_duration * frame_rate)

    samples_per_frame = len(samples) // total_frames

    frames_dir = Path("waveforms") / audio_clip_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []

    for frame_idx in range(total_frames):
        start = frame_idx * samples_per_frame
        end = start + samples_per_frame
        frame_samples = samples[start:end]

        # Plot
        fig, ax = plt.subplots(figsize=(6,10))
        fig.patch.set_facecolor('black')
        ax.plot(frame_samples, color='white')
        ax.axis('off')

        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frame_paths.append(str(frame_path))

    # Create video from frames
    audio_clip = AudioFileClip(str(audio_clip_path))
    video_clip = ImageSequenceClip(frame_paths, fps=frame_rate).set_audio(audio_clip)

    output_path = Path("output") / f"{audio_clip_path.stem}.mp4"
    video_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    print(f"✅ Waveform video saved: {output_path.name}\n")

if __name__ == "__main__":
    main()
