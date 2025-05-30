import csv
from pathlib import Path

# Global column headers for consistent logging
FIELDNAMES = [
    "llm_score",
    "sentiment",
    "length_sec",
    "keyword_hit",
    "text",
    "clip_path",
    "predicted_score",
    "label"
]

def log_clip_data(clip_data, path="clips_dataset.csv"):
    """
    Logs all generated clips (both viral and non-viral) to main CSV.
    """
    file_exists = Path(path).exists()

    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        complete_row = {key: clip_data.get(key, "") for key in FIELDNAMES}
        writer.writerow(complete_row)

def log_viral_clip_only(clip_data, path="viral_clips_dataset.csv"):
    """
    Logs only viral clips (label == 1) to a separate CSV.
    """
    file_exists = Path(path).exists()

    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        complete_row = {key: clip_data.get(key, "") for key in FIELDNAMES}
        writer.writerow(complete_row)
