"""Wildcard query processor for Task 1."""

from typing import List, Set

from index.access import find_wildcard_matches, get_posting_list


def _generate_wildcard_candidates(pattern: str, index_path: str) -> List[str]:
    """Generate candidate vocabulary terms using the wildcard index."""
    # Part A Task 2 provides boundary-padded character n-grams. See the
    # Week 04 lecture sections "k-gram Index" and
    # "Wildcard Expansion Pipeline".
    # TODO(Task 1): generate and combine candidate vocabulary terms.
    raise NotImplementedError


def _filter_wildcard_candidates(
    pattern: str,
    candidates: List[str],
) -> List[str]:
    """Apply the complete wildcard pattern to candidate terms."""
    # This is adapted from the Week 04 workshop function
    # post_filter_wildcard_candidates(...).
    if pattern.endswith("*") and not pattern.startswith("*"):
        prefix = pattern[:-1]
        return sorted(term for term in candidates if term.startswith(prefix))

    if pattern.startswith("*") and not pattern.endswith("*"):
        suffix = pattern[1:]
        return sorted(term for term in candidates if term.endswith(suffix))

    # TODO(Task 1): adapt the workshop helper for an internal wildcard pattern
    # such as learn*ing.
    raise NotImplementedError


def _matching_terms_to_doc_ids(
    matched_terms: List[str],
    index_path: str,
) -> Set[int]:
    """Return documents containing any matched vocabulary term."""
    # This is the matched terms -> posting lists -> document IDs stage in the
    # Week 04 lecture section "Wildcard Expansion Pipeline".
    # TODO(Task 1): combine the posting lists.
    raise NotImplementedError


def process_wildcard_query(pattern: str, index_path: str) -> Set[int]:
    """Evaluate one wildcard query.

    Args:
        pattern: Wildcard pattern with exactly one ``*`` at the start, end,
            or inside the fixed text, such as ``climat*``, ``*ing``, or
            ``learn*ing``. The ``*`` may match any sequence of characters,
            including the empty sequence.
        index_path: Path to the unified index package created by Task 1.

    Returns:
        Set of document IDs containing at least one term matching the pattern.
    """
    # Wildcard overall process:
    # wildcard pattern -> character n-grams -> candidate terms
    # -> full pattern check -> posting lists -> document IDs
    candidates = _generate_wildcard_candidates(pattern, index_path)
    matched_terms = _filter_wildcard_candidates(pattern, candidates)
    return _matching_terms_to_doc_ids(matched_terms, index_path)
