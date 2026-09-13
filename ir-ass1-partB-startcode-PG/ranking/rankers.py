"""Minimal lexical ranking baseline for Task 2.

This module uses a deliberately weak keyword-match baseline so the full system
can run immediately. Keep the function contract unchanged when adding your own
methods.
"""

from typing import List, Sequence, Tuple
from index.io import load

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

# BM25 using our index 
def _rank_bm25(
        query_toks: List[str], 
        candidate_docs: List[List[str]], 
        doc_ids: List[int], 
        inverted_index_path: str, 
        k1: float = 1.2, 
        b: float = 0.75
        ) -> Tuple[List[int], List[float]]:
    """
    # example input from query_toks = ["machine", "learning"]
    # example input from candidate_docs = [ ["machine", "learning", "model"], ["machine", "factory"], ["football", "match"] ]
    # example input from doc_ids = [191, 279, 305] e.g. doc 191 contains "machine", "learning", "model", etc.
    # k1, b are BM25 settings controlling how much repetition saturates and document length affects the score respectively
    # example expected output = ( [191, 279, 305], [3.72. 1.14, 0.0]) e.g. doc 191 has score of 3.72
    # higher score = better. Therefore Rank 1: doc 191 with score 3.72, Rank 2: doc 279 ...
    # break ties with ascending doc_id e.g. if doc 191: 3.72 and doc 279: 3.72 then doc 191 is ranked 1
    """

    package = load(inverted_index_path)

    meta = package["__META__"]
    unified = package["unified"]

    N = meta["N"]
    doc_lengths = meta["doc_lengths"]
    avgdl = meta["avgdl"]

    # each document gets one final score
    scores = []

    # for each query term, extract its corpus df and tf
    for doc_id in doc_ids:
        # running total BM25 score for current document
        running_total_BM25 = 0.0

        for term in query_toks:
            # does query_term even exist in corpus? e.g. ["climate", "afasdfafv"]
            if term not in unified:
                continue

            term_data = unified[term]
            df = term_data["df"]

            # current candidate doc_id may not contain our current evaluated term
            # but the candidate doc_id is here because it contained other terms
            # e.g. candidate doc_id = [10, 20, 30, 50] and term_queries = ["climate", "change", "affects", "poop"]
            # we are currently evaluating "climate" and doc 10 but doc 10 may not have "climate"; it is here because it may contain other term queries like "poop"
            # but if we tried to get doc 10 from "climate"'s posting, we'll get a KeyError: 10
            posting = term_data["postings"].get(doc_id)
            if posting is None:
                continue

            tf = posting["tf"]

            # atp we have everything we need for BM25: N, doc_lengths, avgdl, df, tf, k1, b
            # BM25 formula for one query term in one document: IDF(term) * tf(k1 + 1) / (tf +k1(1 - b + (b * doc_lengths[doc_id]/avgdl)) )
            current_document_length = doc_lengths[doc_id]
            idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))

            # is current document unusually long or short?
            length_norm = 1.0 - b + b * (current_document_length / avgdl)

            # accounting for document length, repeated occurences saturation rate
            tf_component = (tf * (k1 + 1.0) / (tf +k1 * length_norm))

            term_score = idf * tf_component

            running_total_BM25 += term_score

        scores.append((int(doc_id), float(running_total_BM25)))


    # sort scores by descending and doc_id in ascending in case of ties
    # example scores = [(305, 5.40), (191, 3.72), (279, 1.24)]
    scores.sort(key=lambda item: (-item[1], item[0]))

    # but output requires ([305, 191, 279], [5.40, 3.72, 1.24])
    ranked_ids = [doc_id for doc_id, _ in scores]
    ranked_scores = [score for _, score in scores]

    return ranked_ids, ranked_scores

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
    elif method == "bm25":
        return _rank_bm25(query_toks, candidate_docs, doc_ids, inverted_index_path)

    
