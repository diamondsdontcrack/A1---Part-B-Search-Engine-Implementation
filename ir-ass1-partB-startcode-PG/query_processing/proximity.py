"""Proximity query processor for Task 1."""

from typing import List, Set

from query_processing.boolean import process_boolean_query


def _filter_candidates_by_distance(
    query_tokens: List[str],
    candidate_doc_ids: Set[int],
    index_path: str,
) -> Set[int]:
    """Keep candidate documents that satisfy the required NEAR distance."""
    # TODO(Task 1): use the positional index to apply the NEAR/k condition.
    raise NotImplementedError


def process_proximity_query(query: str, index_path: str) -> Set[int]:
    """Evaluate a unigram ``left NEAR/k right`` query.

    Args:
        query: Proximity query such as ``climate NEAR/3 change``.
        index_path: Path to the unified index package created by Task 1.

    Returns:
        Set of document IDs where the two unigram operands occur within
        distance ``k``.

    Semantics:
        Distance is ``abs(left_position - right_position)`` and is
        order-insensitive.
    """
    query_tokens = query.split()

    # Reuse Boolean retrieval for the candidate-document stage.
    and_query = f"{query_tokens[0]} AND {query_tokens[2]}"
    candidate_doc_ids = process_boolean_query(and_query, index_path)

    # Proximity process:
    # candidate document IDs -> position lists -> distance check
    # -> matching document IDs
    #
    # Revisit the Week 04 lecture sections "The NEAR/k Operator",
    # "Positional Index", and "Proximity Trace".
    return _filter_candidates_by_distance(
        query_tokens,
        candidate_doc_ids,
        index_path,
    )
