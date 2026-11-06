import random, hashlib
from datasets import load_dataset
import config

def _sample(ds, n):
    random.seed(config.SEED)
    return random.sample(range(len(ds)), min(n, len(ds)))

def _doc_id(title, text):
    return f"{title}_{hashlib.md5(text.encode()).hexdigest()[:8]}"

def load_hotpotqa(n):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [{"question": ds[i]["question"], "answers": [ds[i]["answer"]],
             "context": ds[i]["context"],
             "gold_titles": set(ds[i]["supporting_facts"]["title"])} for i in _sample(ds, n)]

def load_arcd(n):
    ds = load_dataset("hsseinmz/arcd", split="validation")
    out = []
    for i in _sample(ds, n):
        r = ds[i]; did = _doc_id(r["title"], r["context"])
        out.append({"question": r["question"], "answers": list(dict.fromkeys(r["answers"]["text"])),
                    "context": {"title": [did], "sentences": [[r["context"]]]},
                    "gold_titles": {did}})
    return out

def load_musique(n):
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    ds = ds.filter(lambda r: r["answerable"])
    out = []
    for i in _sample(ds, n):
        r = ds[i]; titles, sents, gold = [], [], set()
        for p in r["paragraphs"]:
            did = _doc_id(p["title"], p["paragraph_text"])
            if did in titles:
                continue
            titles.append(did); sents.append([p["paragraph_text"]])
            if p["is_supporting"]:
                gold.add(did)
        out.append({"question": r["question"],
                    "answers": [r["answer"]] + list(r["answer_aliases"] or []),
                    "context": {"title": titles, "sentences": sents}, "gold_titles": gold})
    return out

LOADERS = {"hotpotqa": load_hotpotqa, "arcd": load_arcd, "musique": load_musique}

def build_corpus(questions):
    corpus = {}
    for q in questions:
        for t, s in zip(q["context"]["title"], q["context"]["sentences"]):
            if t not in corpus:
                corpus[t] = {"title": t, "text": " ".join(s)}
    return list(corpus.values())

def chunk_text(text, size, overlap):
    words = text.split(); chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return chunks

def build_chunks(corpus):
    chunks = []
    for doc in corpus:
        for j, c in enumerate(chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append({"id": f"{doc['title']}_{j}", "title": doc["title"], "text": c})
    return chunks