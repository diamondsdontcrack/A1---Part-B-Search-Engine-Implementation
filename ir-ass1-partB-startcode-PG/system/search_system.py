#!/usr/bin/env python3
"""Batch search-system CLI for Task 3.

Usage:
  python -m system.search_system <queries_json> <documents_jsonl> <run_output_json> [method]

The optional method argument is the Task 3 optimisation selector:
- ``no_optimisation`` is the runnable starter baseline.
- ``default`` is your submitted final system. In the starter, it is initially
  the same as ``no_optimisation``; add your own method branch if you improve it.

The Task 2 ranker method remains ``default`` unless you explicitly extend
the ranking interface elsewhere.
"""

import json
import pathlib
import sys
from typing import Dict, List, Set, Tuple

from index.builders import create_all_indexes
from query_processing.query_process import process_query
from ranking.rankers import rank_documents
from utils.text_preprocessing import preprocess
from nltk.corpus import stopwords
from collections import Counter
import math

_TASK1_FALLBACK_NOTE = (
    "Ranking all documents so Tasks 2 and 3 remain runnable. "
    "This fallback does not show that Task 1 is correct. "
    "Run python -m test_sanity.check_submission and fix any Task 1 failures "
    "before submission."
)
_task1_fallback_warning_shown = False


def _warn_task1_fallback(reason: str) -> None:
    """Show the Task 1 fallback warning once per CLI run."""
    global _task1_fallback_warning_shown
    if _task1_fallback_warning_shown:
        return
    print(f"WARNING: {reason} {_TASK1_FALLBACK_NOTE}")
    _task1_fallback_warning_shown = True


def _load_docs(path: str) -> Tuple[List[dict], List[str]]:
    """Load JSONL documents with ``id`` and ``text`` fields."""
    raw_docs: List[dict] = []
    texts: List[str] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "id" not in obj or "text" not in obj:
                raise ValueError(f"Line {line_num} must contain 'id' and 'text'")
            doc_id = int(obj["id"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            raw_docs.append({"id": doc_id, "text": str(obj["text"])})
            texts.append(str(obj["text"]))

    return raw_docs, texts


def _load_queries(path: str) -> List[dict]:
    """Load query JSON list with ``qid`` and ``query`` fields."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("queries_json must contain a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "qid" not in item or "query" not in item:
            raise ValueError(f"Query item {i} must contain 'qid' and 'query'")
    return data


def _all_doc_candidates(doc_ids: List[int]) -> Set[int]:
    """Return all document IDs."""
    return set(doc_ids)


def _basic_tokenize(text: str) -> List[str]:
    """Tiny baseline tokenizer for the runnable starter."""
    return str(text).split()


def _tokenize_texts(texts: List[str]) -> List[List[str]]:
    """Tokenize with the weak default baseline."""
    return [_basic_tokenize(text) for text in texts]


def _build_index(tokenized_docs: List[List[str]], doc_ids: List[int]) -> Tuple[pathlib.Path, bool]:
    """Build the Task 1 index package used by query processing."""
    cache_dir = pathlib.Path(__file__).resolve().parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    index_path = cache_dir / "unified_package.pkl.gz"

    try:
        if index_path.exists():
            if not index_path.is_file():
                raise IsADirectoryError(f"Index path exists but is not a file: {index_path}")
            index_path.unlink()
        create_all_indexes(tokenized_docs, str(index_path), doc_ids=doc_ids)
        return index_path, True
    except NotImplementedError:
        _warn_task1_fallback("Task 1 index build is not implemented.")
    except Exception as e:
        _warn_task1_fallback(
            f"Task 1 index build failed ({type(e).__name__}: {e})."
        )
    return index_path, False


def _candidate_ids_for_query(
    query_text: str,
    qid: str,
    index_path: pathlib.Path,
    index_ready: bool,
    doc_ids: List[int],
    system_method: str,
) -> Set[int]:
    """Select candidates, using all documents only when Task 1 is unavailable."""
    if system_method not in {
        "no_optimisation",
        "doc_cleaning",
        "query_cleaning",
        "preprocessing",
        "prf",
        "default",
    }:
        raise ValueError(
            f"Unknown system method: {system_method}. "
            "Add a method branch in system/search_system.py."
        )

    if not index_ready:
        return _all_doc_candidates(doc_ids)

    try:
        candidate_ids = process_query(query_text, str(index_path))
    except NotImplementedError:
        _warn_task1_fallback(
            f"Task 1 query processing is not implemented for {qid}."
        )
        candidate_ids = _all_doc_candidates(doc_ids)
    except Exception as e:
        _warn_task1_fallback(
            f"Task 1 query processing failed for {qid} "
            f"({type(e).__name__}: {e})."
        )
        candidate_ids = _all_doc_candidates(doc_ids)

    # A successful empty result means that the query matched no documents.
    return candidate_ids


def _rank_one_query(
    qid: str,
    query_toks: List[str],
    candidate_ids: Set[int],
    doc_to_tokens: Dict[int, List[str]],
    index_path: pathlib.Path,
) -> dict:
    """Rank candidates and format one top-10 run-file result object."""
    candidate_ids_list = sorted(
        int(doc_id)
        for doc_id in candidate_ids
        if int(doc_id) in doc_to_tokens
    )
    candidate_docs = [doc_to_tokens[doc_id] for doc_id in candidate_ids_list]

    ranked_ids, scores = rank_documents(
        query_toks=query_toks,
        candidate_docs=candidate_docs,
        doc_ids=candidate_ids_list,
        inverted_index_path=str(index_path),
        method="default",
    )

    return {
        "qid": qid,
        "doc_ids": [int(doc_id) for doc_id in ranked_ids[:10]],
        "scores": [float(score) for score in scores[:10]],
    }

def _is_structured_query(query_text: str) -> bool:
    """
    Expected input:
        query_text: Raw query string.

    Expected output:
        True if the query contains structured syntax such as Boolean operators,
        parentheses, wildcard '*' or NEAR/k. Otherwise False.

    Structured queries are detected so their special syntax is not changed
    by normal text preprocessing. It needs to be preserved for Task 1's boolean.py
    """
    tokens = _basic_tokenize(query_text)

    is_structured = any(
        token in {"AND", "OR", "NOT", "(", ")"}
        or token.startswith("NEAR/")
        or "*" in token
        for token in tokens
    )

    return is_structured

def _clean_query_tokens(query_text: str) -> List[str]:
    """
    Expected input:
        query_text: Raw query string.

    Expected output:
        List of query tokens.

    Structured queries keep their syntax unchanged, while natural-language
    queries use the same preprocessing as the document collection.
    """
    if _is_structured_query(query_text):
        return _basic_tokenize(query_text)

    return preprocess([query_text])[0]

_STOPWORDS = set(stopwords.words("english"))

def _document_frequencies(
    tokenized_docs: List[List[str]]
) -> Counter:
    """
    Expected input:
        tokenized_docs: List of tokenised documents.

    Expected output:
        Counter mapping each term to the number of documents that contain it.

    Document frequency is later used to calculate IDF when selecting
    PRF expansion terms.
    """
    df = Counter()

    for doc in tokenized_docs:
        # set(doc) ensures a term contributes at most once per document
        df.update(set(doc))

    return df

def _select_prf_terms(
    top_doc_ids: List[int],
    doc_to_tokens: Dict[int, List[str]],
    query_toks: List[str],
    doc_freq: Counter,
    num_docs: int,
    top_m: int,
) -> List[str]:
    """
    Expected input:
        top_doc_ids: IDs of the documents assumed to be pseudo-relevant.
        doc_to_tokens: Maps each document ID to its token list.
        query_toks: Tokens already present in the original query.
        doc_freq: Corpus document frequency for each term.
        num_docs: Total number of documents in the collection.
        top_m: Number of expansion terms to return.

    Expected output:
        List of the top_m highest TF-IDF terms selected for query expansion.

    Terms that are punctuation, stopwords or already in the query are ignored.
    """

    # count term frequency across the pseudo-relevant documents.
    term_counts = Counter()

    for doc_id in top_doc_ids:
        term_counts.update(doc_to_tokens[doc_id])

    query_terms = set(query_toks)
    scored_terms = []

    for term, tf in term_counts.items():

        # ignore punctuation/numeric tokens
        if not term.isalpha():
            continue

        # common words such as "the" and "is" are poor expansion terms.
        if term in _STOPWORDS:
            continue

        # don't add words already present in the original query.
        if term in query_terms:
            continue

        df = doc_freq[term]

        # rare corpus terms receive higher IDF.
        idf = math.log((num_docs + 1) / (df + 1))

        # pPrefer terms frequent in the feedback docs but uncommon globally.
        score = tf * idf

        scored_terms.append((term, score))

    # highest TF-IDF first, then alphabetical order for deterministic ties.
    scored_terms.sort(
        key=lambda item: (-item[1], item[0])
    )

    return [
        term
        for term, _ in scored_terms[:top_m]
    ]

def main() -> int:
    if len(sys.argv) not in (4, 5):
        print("Usage: python -m system.search_system <queries_json> <documents_jsonl> <run_output_json> [method]")
        return 2

    queries_path, docs_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    system_method = sys.argv[4] if len(sys.argv) == 5 else "default"
    if system_method not in {
        "no_optimisation",
        "doc_cleaning",
        "query_cleaning",
        "preprocessing",
        "prf",
        "default",
    }:
        raise ValueError(
            f"Unknown system method: {system_method}. "
            "Use 'no_optimisation' or 'default', or add a new method branch."
        )
    # Task 3 implementation:
    # no_optimisation -> provided runnable pipeline
    # default -> the same pipeline + an adapted non-ranking optimisation
    #
    # The Week 02 workshop provides preprocessing variants. The Week 05
    # workshop provides a query-expansion activity. The Week 06 workshop
    # provides controlled pipeline comparisons. Your completed Part A
    # preprocessing may also be adapted here.
    #
    # Add the optimisation at the stage where it naturally belongs. Modify the
    # existing pipeline rather than copying the full pipeline. Keep
    # no_optimisation unchanged.
    #
    # If you preprocess a structured query, preserve syntax-bearing tokens
    # before general cleaning and restore them before query processing.
    # Preserve Boolean operators such as AND, OR, and NOT, parentheses such as
    # ( climate OR policy ), wildcard syntax such as learn*ing, and proximity
    # syntax such as NEAR/3.

    queries = _load_queries(queries_path)
    raw_docs, doc_texts = _load_docs(docs_path)

    if system_method == "no_optimisation":
        tokenized_docs = _tokenize_texts(doc_texts)

    # use my part A text_preprocess.py
    elif system_method == "doc_cleaning":
        tokenized_docs = preprocess(doc_texts)

    # query cleaning only, documents untouched
    elif system_method == "query_cleaning":
        tokenized_docs = _tokenize_texts(doc_texts)

    # for final chosen optimisation method
    elif system_method in {"preprocessing", "prf", "default"}:
        tokenized_docs = preprocess(doc_texts)

    doc_ids = [int(doc["id"]) for doc in raw_docs]
    doc_to_tokens: Dict[int, List[str]] = {
        doc_id: toks
        for doc_id, toks in zip(doc_ids, tokenized_docs)
    }

    doc_freq = _document_frequencies(tokenized_docs)

    index_path, index_ready = _build_index(tokenized_docs, doc_ids)

    results = []
    for item in queries:
        qid = str(item["qid"])
        query_text = str(item["query"])

        if system_method in {"query_cleaning", "preprocessing", "prf", "default"}:
            query_toks = _clean_query_tokens(query_text)
        else:
            query_toks = _basic_tokenize(query_text)

        query_for_processing = " ".join(query_toks)
        candidate_ids = _candidate_ids_for_query(
            query_text=query_for_processing,
            qid=qid,
            index_path=index_path,
            index_ready=index_ready,
            doc_ids=doc_ids,
            system_method=system_method,
        )

        if system_method in {"prf", "default"} and not _is_structured_query(query_text):


            # first search (normal search using OG query)
            initial_result = _rank_one_query(qid, query_toks, candidate_ids, doc_to_tokens, index_path,)

            # assume top 3 initially searched docs are relevant
            top_doc_ids = initial_result["doc_ids"][:3]

            # extract the top N strongest TF-IDF terms from the above documents
            expansion_terms = _select_prf_terms(
                top_doc_ids=top_doc_ids,
                doc_to_tokens=doc_to_tokens,
                query_toks=query_toks,
                doc_freq=doc_freq,
                num_docs=len(tokenized_docs),
                top_m=20,
            )

            # add feedback terms while preserving all OG query terms
            expanded_query_toks = query_toks + expansion_terms
            expanded_query_text = " ".join(expanded_query_toks)

            # second retrieval (search again using the expanded query)
            candidate_ids = _candidate_ids_for_query(
                query_text=expanded_query_text,
                qid=qid,
                index_path=index_path,
                index_ready=index_ready,
                doc_ids=doc_ids,
                system_method=system_method,
            )

            # second ranking using BM25
            results.append(
                _rank_one_query(
                    qid,
                    expanded_query_toks,
                    candidate_ids,
                    doc_to_tokens,
                    index_path,
                )
            )

        else:
            results.append(_rank_one_query(qid, query_toks, candidate_ids, doc_to_tokens, index_path))

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} query results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
