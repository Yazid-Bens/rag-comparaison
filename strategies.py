import config
from llm import call_llm, parse_json

ANSWER_PROMPT = """Answer the question using ONLY the context below.
Reply with a short answer (a few words at most). Do not explain.
If the context is insufficient, reply "I don't know".
Context:
{context}
Question: {question}
Answer:"""

def build_context(chunks):
    return "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)

def answer(question, chunks):
    return call_llm(ANSWER_PROMPT.format(context=build_context(chunks), question=question), max_tokens=60)

class Naive:
    def run(self, q, idx):
        return [idx.chunks[i] for i, _ in idx.dense_search(q, config.TOP_K)]

class Hybrid:
    def run(self, q, idx):
        return [idx.chunks[i] for i in idx.hybrid_search(q, config.TOP_K, pool=10)]

class HyDE:
    def run(self, q, idx):
        hypo = call_llm(f"Write a short factual passage answering: {q}", 0.4)
        return [idx.chunks[i] for i, _ in idx.dense_search(hypo, config.TOP_K)]

class QueryExpansion:
    def run(self, q, idx):
        rw = call_llm(f"Rewrite this question in 3 different ways, one per line:\n{q}")
        pool = {}
        for r in [q] + [x for x in rw.split("\n") if x.strip()]:
            for i, _ in idx.dense_search(r, 5):
                pool[i] = pool.get(i, 0) + 1
        top = sorted(pool, key=pool.get, reverse=True)[:config.TOP_K]
        return [idx.chunks[i] for i in top]

class Corrective:
    def run(self, q, idx):
        cands = [idx.chunks[i] for i, _ in idx.dense_search(q, 10)]
        good, rest = [], []
        for c in cands:
            v = call_llm(f"Question: {q}\nPassage: {c['text']}\nCould this passage help answer the question, "
                         f"directly or by giving an intermediate fact about a related entity? Reply YES or NO.", max_tokens=5)
            (good if v.strip().upper().startswith("YES") else rest).append(c)
        return (good + rest)[:config.TOP_K]

class GraphRAG:
    def run(self, q, idx):
        seeds = [i for i, _ in idx.dense_search(q, 3)]
        ids = (seeds + idx.graph_expand(seeds, config.TOP_K))[:config.TOP_K]
        return [idx.chunks[i] for i in ids]

class AgenticV1:
    def run(self, q, idx):
        context, sub_q = [], q
        for _ in range(3):
            hits = [idx.chunks[i] for i, _ in idx.dense_search(sub_q, 3)]
            context += [c for c in hits if c not in context]
            check = call_llm(f"Context:\n{build_context(context)}\nQuestion: {q}\nIs the context enough? Reply YES or NO.", max_tokens=5)
            if "YES" in check.upper() or len(context) >= config.TOP_K:
                break
            sub_q = call_llm(f"Write one follow-up search query to find missing info for: {q}")
        return context[:config.TOP_K]

REFLECT_PROMPT = """You are a research agent answering a multi-hop question by searching a document collection.

Question: {q}

Evidence notes so far:
{notes}

Newly retrieved passages:
{cands}

Do 3 things:
1. "useful": numbers of the passages that contain information useful for answering the question, including intermediate facts about entities needed to reach the answer. Empty list if none.
2. "notes": updated evidence notes, at most 3 short sentences, keeping only facts that matter for the question.
3. "sufficient": true only if the notes already contain everything needed to answer. Otherwise false, and give "next_query": a short, specific search query for the single most important missing fact (use entity names found in the notes; write it in the same language as the question).

Reply with JSON only: {{"useful":[1],"notes":"...","sufficient":false,"next_query":"..."}}"""

class Agentic:
    def run(self, q, idx):
        K = config.TOP_K
        seen, evidence = set(), []
        notes, query, prev = "(none yet)", q, set()
        for step in range(config.MAX_AGENT_STEPS):
            prev.add(query)
            ranked = [i for i in idx.hybrid_search(query, 25, pool=30) if i not in seen]
            hits = ranked[:config.CANDIDATES_PER_STEP]
            if not hits:
                break
            seen.update(hits)
            cands = "\n".join(f"[{n}] {idx.chunks[i]['text'][:450]}" for n, i in enumerate(hits, 1))
            data = parse_json(call_llm(REFLECT_PROMPT.format(q=q, notes=notes, cands=cands),
                                       max_tokens=300, json_mode=True))
            if not isinstance(data, dict):
                evidence += [(step, r, i) for r, i in enumerate(hits[:2])]
                break
            useful = data.get("useful", [])
            for n in (useful if isinstance(useful, list) else []):
                try:
                    pos = int(n) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= pos < len(hits):
                    evidence.append((step, pos, hits[pos]))
            notes = str(data.get("notes") or notes)[:600]
            nxt = str(data.get("next_query") or "").strip()
            n_ev = len({e[2] for e in evidence})
            if data.get("sufficient") is True and (n_ev >= 2 or step >= 1):
                break
            if not nxt or nxt in prev:
                nxt = (q + " " + notes)[:300]
                if nxt in prev:
                    break
            query = nxt
        uniq = {}
        for step, r, i in evidence:
            uniq.setdefault(i, (step, r))
        chosen = sorted(uniq.items(), key=lambda kv: (kv[1][1], kv[1][0]))[:K]
        chosen.sort(key=lambda kv: kv[1])
        ids = [i for i, _ in chosen]
        if len(ids) < K:
            for i in idx.hybrid_search(q, K * 3, pool=30):
                if i not in ids:
                    ids.append(i)
                if len(ids) == K:
                    break
        return [idx.chunks[i] for i in ids]

class Adaptive:
    def run(self, q, idx):
        route = call_llm(f"Question: {q}\nDoes answering require combining facts from more than one document? "
                         f"Reply COMPLEX if yes, SIMPLE if no. One word.", max_tokens=5)
        if "COMPLEX" in route.upper():
            return Agentic().run(q, idx)
        return Naive().run(q, idx)

STRATEGIES = {
    "Naive": Naive, "Hybrid": Hybrid, "HyDE": HyDE, "QueryExpansion": QueryExpansion,
    "Corrective": Corrective, "GraphRAG": GraphRAG, "Agentic_v1": AgenticV1,
    "Agentic": Agentic, "Adaptive": Adaptive,
}