import json, re
import ollama
import config

_client = ollama.Client(timeout=180)
STATS = {"calls": 0}

def call_llm(prompt, temperature=config.TEMPERATURE, max_tokens=200, json_mode=False):
    STATS["calls"] += 1
    extra = {"format": "json"} if json_mode else {}
    try:
        r = _client.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"num_ctx": config.NUM_CTX, "temperature": temperature,
                     "num_predict": max_tokens, "repeat_penalty": 1.1},
            keep_alive=-1, **extra)
        return r.message.content.strip()
    except Exception as e:
        print(f"[LLM error] {type(e).__name__}: {e}")
        return ""

def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None