from moviepy.editor import VideoFileClip
from pathlib import Path
from .logger import log_clip_data, log_viral_clip_only



def extract_audio_from_video(video_path: Path, audio_output_path: Path):
    print(f"🎥 Extracting audio from video: {video_path.name}")
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_output_path), codec='pcm_s16le')  # Save as WAV
    print(f"✅ Saved audio to: {audio_output_path.name}")
    return audio_output_path

def cut_highlight_video_clips(video_path: Path, highlights, virality_model, output_dir=Path("output")):
    print(f"✂️ Cutting video highlights from: {video_path.name}")
    video = VideoFileClip(str(video_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, h in enumerate(highlights):
        start = h["start"]
        end = h["end"]
        clip = video.subclip(start, end)

        out_path = output_dir / f"{video_path.stem}_clip_{idx+1}.mp4"
        clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac")
        print(f"✅ Saved highlight: {out_path.name}")

        # Predict label
        features = [[
            h["llm_score"],
            h["sentiment"],
            h["length_sec"],
            int(h["keyword_hit"])
        ]]
        label = int(virality_model.predict(features)[0])

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

