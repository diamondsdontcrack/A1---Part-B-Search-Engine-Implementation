"""Proximity query processor for Task 1."""

from typing import List, Set

from index.access import get_term_positions
from query_processing.boolean import process_boolean_query


def _filter_candidates_by_distance(
    query_tokens: List[str],
    candidate_doc_ids: Set[int],
    index_path: str,
) -> Set[int]:
    """Keep candidate documents that satisfy the required NEAR distance."""
    # TODO(Task 1): use the positional index to apply the NEAR/k condition.

    # example input: climate NEAR/2 policy 
    # process_proximity_query's query_tokens = ["climate", "NEAR/2", "policy"]
    # this function takes in candidate_doc_ids = {10, 20}
    # some rules: climate NEAR/2 climate, both climate must be from different occurences
    # only one valid pair is required

    left_word = query_tokens[0]
    right_word = query_tokens[2]
    k = int(query_tokens[1].split("/")[1])
    final_set: Set[int] = set()

    for doc_id in candidate_doc_ids:
        # get positions of left and right word for current document
        left_positions = get_term_positions(left_word, doc_id, index_path)
        right_positions = get_term_positions(right_word, doc_id, index_path)

        # improved version (v2) using a variation of 2 pointer algorithm instead of a triple nested for loop.
        # when encountering case climate NEAR/k climate
        if left_word == right_word: # e.g left_positions = [2, 7, 8, 15] means same document has 4 distinct "climate"
            for i in range(len(left_positions) - 1):
                # 
                if left_positions[i+1] - left_positions[i] <= k:
                    final_set.add(doc_id)
                    break
            continue

        # when encountering any other case
        i,j = 0, 0

        while i < len(left_positions) and j < len(right_positions):
            left_pos = left_positions[i]
            right_pos = right_positions[j]

            # abs(left_word position - right_word position) <= k then all good
            if abs(left_pos - right_pos) <= k:
                final_set.add(doc_id)
                break

            # move whichever position is further behind
            if left_pos < right_pos:
                i += 1
            else:
                j += 1

    return final_set

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
