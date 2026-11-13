"""Local cybersecurity knowledge base B and retrieval for RAG (Section III-C).

Entries are JSON objects {"id", "category", "title", "text"} stored in
data/kb/*.json. Retrieval uses TF-IDF cosine similarity, so no embedding
model or internet connection is needed on the 8 GB host. The query combines
the evidence label c_i with simple descriptors of the flow features x_i.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .features import FEATURE_NAMES

PORT_HINTS = {22: "ssh port 22 login authentication", 80: "http web port 80",
              443: "https tls port 443", 21: "ftp port 21", 23: "telnet port 23"}


class KnowledgeBase:
    def __init__(self, kb_dir: str, top_k: int = 3) -> None:
        self.entries: List[Dict] = []
        for path in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
            with open(path) as f:
                data = json.load(f)
            self.entries.extend(data if isinstance(data, list) else [data])
        if not self.entries:
            raise ValueError(f"no knowledge-base entries found in {kb_dir}")
        self.top_k = top_k
        docs = [f"{e['category']} {e['title']} {e['text']}" for e in self.entries]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(docs)

    @staticmethod
    def describe(x: Sequence[float]) -> str:
        f = dict(zip(FEATURE_NAMES, x))
        words = [PORT_HINTS.get(int(f["dst_port"]), f"port {int(f['dst_port'])}")]
        if f["src_unique_dst_ports"] > 20:
            words.append("many destination ports scanning")
        if f["pkts_per_s"] > 500:
            words.append("high packet rate flood volume")
        if f["syn_count"] > 0 and f["ack_count"] == 0:
            words.append("syn without ack half-open")
        return " ".join(words)

    def search(self, query: str) -> List[Dict]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][: self.top_k]
        return [self.entries[i] for i in order]

    def retrieve(self, category: str, x: Sequence[float]) -> str:
        """Return the retrieved context C as text for the LLM prompt."""
        query = f"{category.replace('_', ' ').replace('-', ' ')} {self.describe(x)}"
        return "\n".join(f"- {e['title']}: {e['text']}" for e in self.search(query))
