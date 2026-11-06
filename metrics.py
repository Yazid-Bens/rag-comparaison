import re, unicodedata
from collections import Counter
from llm import call_llm

def normalize(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return s.split()

def _f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return 0.0
    num = sum((Counter(p) & Counter(g)).values())
    if num == 0: return 0.0
    pr, rc = num / len(p), num / len(g)
    return 2 * pr * rc / (pr + rc)

def _em(pred, gold):
    p, g = normalize(pred), normalize(gold)
    return float(bool(p) and p == g)

def _as_list(g): return [g] if isinstance(g, str) else list(g)
def f1(pred, golds): return max(_f1(pred, g) for g in _as_list(golds))
def exact_match(pred, golds): return max(_em(pred, g) for g in _as_list(golds))

def recall_at_k(retrieved, gold, k):
    return len({c["title"] for c in retrieved[:k]} & gold) / max(len(gold), 1)

def precision_at_k(retrieved, gold, k):
    return len({c["title"] for c in retrieved[:k]} & gold) / k

def mrr(retrieved, gold):
    for r, c in enumerate(retrieved, 1):
        if c["title"] in gold: return 1.0 / r
    return 0.0

def _score(text):
    m = re.search(r"\d*\.?\d+", text or "")
    return max(0.0, min(1.0, float(m.group()))) if m else 0.0

def context_precision(question, retrieved):
    if not retrieved: return 0.0
    hits = 0
    for c in retrieved:
        v = call_llm(f"Is this passage useful to answer the question?\nQuestion: {question}\n"
                     f"Passage: {c['text'][:500]}\nReply YES or NO.", max_tokens=5)
        hits += "YES" in v.upper()
    return hits / len(retrieved)

def faithfulness(question, answer_text, retrieved):
    if not retrieved: return 0.0
    ctx = "\n".join(c["text"][:300] for c in retrieved)
    return _score(call_llm(f"Context:\n{ctx}\n\nAnswer: {answer_text}\n\nWhat fraction of the claims in the "
                           f"answer are supported by the context? Reply with only a number between 0 and 1.", max_tokens=8))

def answer_relevancy(question, answer_text):
    return _score(call_llm(f"Question: {question}\nAnswer: {answer_text}\n\nHow well does the answer address "
                           f"the question? Reply with only a number between 0 and 1.", max_tokens=8))