import csv
from pathlib import Path

# Always point to backend/ directory regardless of working directory
BASE_DIR = Path(__file__).resolve().parent.parent
CLIPS_PATH = BASE_DIR / "clips_dataset.csv"
VIRAL_PATH = BASE_DIR / "viral_clips_dataset.csv"

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

def log_clip_data(clip_data, path=CLIPS_PATH):
    """
    Logs metadata for a single generated clip into a CSV file.
    """
    file_exists = Path(path).exists()

    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        complete_row = {key: clip_data.get(key, "") for key in FIELDNAMES}
        writer.writerow(complete_row)

def log_viral_clip_only(clip_data, path=VIRAL_PATH):
    """
    Logs only viral clips (label==1) into a separate CSV file.
    """
    file_exists = Path(path).exists()

    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        complete_row = {key: clip_data.get(key, "") for key in FIELDNAMES}
        writer.writerow(complete_row)
