import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import spacy, networkx as nx
import config

class Indexes:
    def __init__(self, chunks, embed_model=config.EMBED_MODEL):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb = SentenceTransformer(embed_model, device=self.device)
        self.dense = self.emb.encode(self.texts, normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=64)
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = self._build_graph()

    def dense_search(self, query, k):
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        s = self.dense @ q
        return [(int(i), float(s[i])) for i in np.argsort(s)[::-1][:k]]

    def bm25_search(self, query, k):
        s = self.bm25.get_scores(query.lower().split())
        return [(int(i), float(s[i])) for i in np.argsort(s)[::-1][:k]]

    def hybrid_search(self, query, k, pool=10):
        scores = {}
        for lst in (self.dense_search(query, pool), self.bm25_search(query, pool)):
            for r, (i, _) in enumerate(lst):
                scores[i] = scores.get(i, 0) + 1 / (60 + r + 1)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def _build_graph(self):
        g = nx.Graph()
        for i, c in enumerate(self.chunks):
            for e in self.nlp(c["text"]).ents:
                g.add_edge(e.text, i)
        return g

    def graph_expand(self, seed_ids, k):
        extra = []
        for i in seed_ids:
            for e in self.nlp(self.chunks[i]["text"]).ents:
                if e.text in self.graph:
                    for nb in self.graph.neighbors(e.text):
                        if isinstance(nb, int) and nb not in seed_ids and nb not in extra:
                            extra.append(nb)
        return extra[:k]