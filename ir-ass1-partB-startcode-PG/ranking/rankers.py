"""Minimal lexical ranking baseline for Task 2.

This module uses a deliberately weak keyword-match baseline so the full system
can run immediately. Keep the function contract unchanged when adding your own
methods.
"""

from typing import List, Sequence, Tuple

import numpy as np


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using NumPy."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    norm_x = float(np.linalg.norm(x))
    norm_y = float(np.linalg.norm(y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return float(np.dot(x, y) / (norm_x * norm_y))


def _score_binary_match(query_toks: Sequence[str], doc_toks: Sequence[str]) -> float:
    """Return 1.0 if the query and document share any token, else 0.0."""
    if not query_toks or not doc_toks:
        return 0.0
    return 1.0 if (set(query_toks) & set(doc_toks)) else 0.0


def _rank_keyword_match(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
) -> Tuple[List[int], List[float]]:
    """Weak runnable baseline: any shared query token gets score 1.0."""
    scored = []
    for doc_id, doc_toks in zip(doc_ids, candidate_docs):
        scored.append((int(doc_id), _score_binary_match(query_toks, doc_toks)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    ranked_ids = [doc_id for doc_id, _ in scored]
    scores = [float(score) for _, score in scored]
    return ranked_ids, scores


def rank_documents(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    inverted_index_path: str,
    method: str = "default",
) -> Tuple[List[int], List[float]]:
    """Rank every candidate document.

    Args:
        query_toks: Tokenized query terms.
        candidate_docs: Tokenized candidate documents aligned with ``doc_ids``.
        doc_ids: Candidate document IDs.
        inverted_index_path: Path to the unified index package.
        method: Method selector. ``default`` is an alias set inside this
            function.
            Add extra ``elif`` branches using real method names for experiments,
            then point ``best_method`` to your best-performing final method.

    Returns:
        ``(ranked_doc_ids, scores)`` for all candidate docs. ``ranked_doc_ids``
        must be a permutation of the input ``doc_ids``.
    """
    if len(candidate_docs) != len(doc_ids):
        raise ValueError("candidate_docs and doc_ids must have the same length")

    # Task 2 overall process:
    # query tokens + candidate documents -> one score per candidate
    # -> ranked document IDs + aligned scores
    #
    # TF-IDF/vector route:
    # You may adapt your Part A Task 3 TF-IDF and Task 4 semantic-vector
    # implementations for this route.
    # See the Week 03 workshop sections "Implement TF-IDF", "Implement GloVe
    # mean pooling", and "Cosine similarity and ranking comparison". The
    # cosine helper compares two completed representations. You still construct
    # query and document representations, score every candidate, sort all
    # documents, and connect the selected method to default.
    #
    # BM25 route:
    # The unified index uses the Week 05 term-statistics structure: each term
    # stores df and per-document tf. ``index.io.load(inverted_index_path)``
    # loads the package. See the Week 05 workshop sections "Inspect
    # index-stored statistics" and "Implement BM25 scoring". Adapt that work
    # to this ranking contract if you choose BM25.
    #
    # Adapt one or more routes to this candidate-ranking contract. Return every
    # candidate, keep scores aligned with document IDs, and make best_method
    # identify the method represented by "default".

    # Before final submission, set this to the real method name that performs
    # best on the dev evaluation, for example: "tfidf" or "bm25".
    best_method = "keyword_match"

    if method == "default":
        method = best_method

    if method == "keyword_match":
        # TODO(Task 2): replace this baseline with your best ranker if your
        # improved method performs better on the dev evaluation. Keep the
        # method branch names descriptive, then update best_method above.
        return _rank_keyword_match(query_toks, candidate_docs, doc_ids)

    # TODO(Task 2): add optional experimental branches here, for example:
    # elif method == "tfidf":
    #     return ...

    raise ValueError(f"Unknown ranking method: {method}")
