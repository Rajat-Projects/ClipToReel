import ollama
import re
from textblob import TextBlob

important_keywords = [
    "success", "failure", "motivation", "mindset", "discipline", "growth", "focus",
    "confidence", "leadership", "purpose", "overcome", "believe"
]

def llm_score_local(text, model="mistral"):  # ✅ use mistral here
    prompt = f"""
Rate this podcast segment from 1 to 10 based on how engaging and impactful it would be as a social media highlight reel.

Criteria:
- Is it emotionally engaging or motivational?
- Does it contain a strong message or insight?
- Would it make someone stop scrolling in the first 5 seconds?
- Does it feel complete or coherent on its own?

Segment:
\"\"\"{text}\"\"\"
Reply with ONLY a number between 1 and 10.
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()

        # ✅ New: extract number using regex
        match = re.search(r'\b([1-9]|10)\b', content)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"⚠️ LLM fallback triggered due to: {e}")

    # === Fallback scoring ===
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    keyword_hit = any(k in text.lower() for k in important_keywords)
    length_score = len(text.split()) / 20  # favors ~20+ word segments

    fallback_score = 5 + (2 * sentiment) + (1 if keyword_hit else 0) + (0.3 * length_score)
    return round(min(max(fallback_score, 1), 10))