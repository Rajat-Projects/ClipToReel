def compute_virality_score(llm_score, sentiment, keyword_hit, length_sec):
    raw_score = (
        (llm_score / 10) * 40 +         # LLM score → max 40 pts
        (sentiment + 1) * 20 +          # Sentiment [-1,1] → [0,2] → max 40 pts
        (15 if keyword_hit else 0) +    # Keyword match → 15 pts
        min(length_sec / 15, 1.5) * 25  # Good if >=15s → max 25 pts
    )
    return min(round(raw_score), 100)