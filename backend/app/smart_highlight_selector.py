"""import json
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from .llm_ranker import llm_score_local
from .scoring import compute_virality_score

nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

important_keywords = ["success", "growth", "failure", "habit", "mindset", "motivation", "leadership", "achievement"]

def smart_highlight_selector(transcript_path, min_clip_duration=20, max_clip_duration=90, top_n=5):
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    segments = transcript["segments"]
    segment_texts = [seg['text'] for seg in segments]
    embeddings = sbert.encode(segment_texts, convert_to_tensor=True)

    merged = []
    used = set()

    for i, seg in enumerate(segments):
        if i in used:
            continue

        group = [seg]
        used.add(i)
        start_time = seg['start']
        end_time = seg['end']
        duration = end_time - start_time

        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > 0.6:
                group.append(segments[j])
                used.add(j)
                end_time = segments[j]['end']
                duration = end_time - start_time
                if duration >= max_clip_duration:
                    break

        if min_clip_duration <= duration <= max_clip_duration:
            combined_text = " ".join([g['text'] for g in group])
            sentiment = TextBlob(combined_text).sentiment.polarity
            keyword_hit = any(kw in combined_text.lower() for kw in important_keywords)
            length_score = len(combined_text.split()) / 20
            #score = sentiment + 0.2 * length_score + (1 if keyword_hit else 0)
            llm_score = llm_score_local(combined_text)
            print(f"📊 Final Clip → Duration: {round(duration, 1)}s | LLM Score: {llm_score}")
            length_sec = round(end_time - start_time, 2)
            virality_score = compute_virality_score(llm_score, sentiment, keyword_hit, length_sec)

            merged.append({
                "start": start_time,
                "end": end_time,
                "text": combined_text,
                "llm_score": llm_score,
                "sentiment": sentiment,
                "keyword_hit": keyword_hit,
                "length_sec": length_sec,
                "virality_score": virality_score
                })

    top_highlights = sorted(merged, key=lambda x: x['llm_score'], reverse=True)[:top_n]

    #top_highlights = sorted(merged, key=lambda x: x['score'], reverse=True)[:top_n]
    print(f"\n✅ Selected {len(top_highlights)} smart highlights (SBERT merged, duration constrained)\n")
    return top_highlights 

"""
import json
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import whisper

from .llm_ranker import llm_score_local
from .scoring import compute_virality_score

# Load NLP and SBERT models
nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Define folder for saving transcripts
TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

important_keywords = ["success", "growth", "failure", "habit", "mindset", "motivation", "leadership", "achievement"]

def smart_highlight_selector(audio_path, min_clip_duration=20, max_clip_duration=90, top_n=5):
    # ----- STEP 1: Transcribe with Whisper if needed -----
    transcript_path = TRANSCRIPT_DIR / f"{audio_path.stem}.json"

    if not transcript_path.exists():
        print(f"📝 Transcribing audio using Whisper model 'base'...")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved transcript to: {transcript_path.name}")
    else:
        print(f"📄 Found existing transcript: {transcript_path.name}")

    # ----- STEP 2: Load transcript -----
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    segments = transcript["segments"]
    segment_texts = [seg['text'] for seg in segments]
    embeddings = sbert.encode(segment_texts, convert_to_tensor=True)

    merged = []
    used = set()

    for i, seg in enumerate(segments):
        if i in used:
            continue

        group = [seg]
        used.add(i)
        start_time = seg['start']
        end_time = seg['end']
        duration = end_time - start_time

        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > 0.6:
                group.append(segments[j])
                used.add(j)
                end_time = segments[j]['end']
                duration = end_time - start_time
                if duration >= max_clip_duration:
                    break

        if min_clip_duration <= duration <= max_clip_duration:
            combined_text = " ".join([g['text'] for g in group])
            sentiment = TextBlob(combined_text).sentiment.polarity
            keyword_hit = any(kw in combined_text.lower() for kw in important_keywords)
            llm_score = llm_score_local(combined_text)
            length_sec = round(end_time - start_time, 2)
            virality_score = compute_virality_score(llm_score, sentiment, keyword_hit, length_sec)

            merged.append({
                "start": start_time,
                "end": end_time,
                "text": combined_text,
                "llm_score": llm_score,
                "sentiment": sentiment,
                "keyword_hit": keyword_hit,
                "length_sec": length_sec,
                "virality_score": virality_score
            })

    top_highlights = sorted(merged, key=lambda x: x['llm_score'], reverse=True)[:top_n]
    print(f"\n✅ Selected {len(top_highlights)} smart highlights (SBERT merged, duration constrained)\n")
    return top_highlights
