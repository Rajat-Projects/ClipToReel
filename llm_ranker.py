import ollama

def llm_score_local(text, model="phi"):
    prompt = f"Rate the following podcast segment from 1 to 10 based on how emotionally impactful or motivational it would be as a short social media clip:\n\n\"{text}\"\n\nOnly respond with a number between 1 and 10."

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response['message']['content'].strip()
    try:
        score = int(content)
        if 1 <= score <= 10:
            return score
    except:
        pass

    return 5  # default if something fails