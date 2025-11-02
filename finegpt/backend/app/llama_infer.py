import requests

API_KEY = "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"

def generate_response(prompt: str):
    """
    Generate a response using Meta Llama 3.3-70B via OpenRouter API.
    The URL is hidden internally; only the key is visible to the user.
    """
    # The URL is internal — you don't need to configure or pass it.
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }

    try:
        response = requests.post(_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Llama API call failed: {e}"
