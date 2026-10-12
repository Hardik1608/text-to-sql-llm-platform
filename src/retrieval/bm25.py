"""
Okapi BM25 with Lucene's non-negative IDF and an inverted index.

    idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

The classic Robertson IDF, ln((N - n_t + 0.5) / (n_t + 0.5)), is zero or negative for terms
that occur in half or more of the documents. On small corpora — a catalog of two databases,
a schema with four tables — that silently zeroes every score and retrieval degenerates to
"return the first document". Lucene/Elasticsearch use the shifted IDF above, which is always
positive. Scoring walks the postings of the query terms only, so cost scales with matches,
not with corpus size.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.n_docs = len(corpus)
        doc_len = np.array([len(doc) for doc in corpus], dtype=float)
        avgdl = float(doc_len.mean()) if self.n_docs and doc_len.mean() > 0 else 1.0
        self._norm = k1 * (1 - b + b * doc_len / avgdl)

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self._postings = {
            t: (np.array(ids), np.array(tfs, dtype=float)) for t, (ids, tfs) in postings.items()
        }
        self._idf = {
            t: math.log(1 + (self.n_docs - len(ids) + 0.5) / (len(ids) + 0.5))
            for t, (ids, _) in self._postings.items()
        }

    def get_scores(self, query: list[str]) -> np.ndarray:
        """BM25 score of every document; documents sharing no term with the query score 0."""
        scores = np.zeros(self.n_docs)
        for term in query:
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            scores[ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + self._norm[ids])
        return scores
