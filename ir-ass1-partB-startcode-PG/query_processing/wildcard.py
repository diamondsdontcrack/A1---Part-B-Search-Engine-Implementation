"""Wildcard query processor for Task 1."""

from typing import List, Set

from index.access import find_wildcard_matches, get_posting_list


def _generate_wildcard_candidates(pattern: str, index_path: str) -> List[str]:
    """Generate candidate vocabulary terms using the wildcard index."""
    # Part A Task 2 provides boundary-padded character n-grams. See the
    # Week 04 lecture sections "k-gram Index" and
    # "Wildcard Expansion Pipeline".
    # TODO(Task 1): generate and combine candidate vocabulary terms.
    
    # say user searches for "learn*ing", prefix will be "learn" and suffix will be "ing"
    prefix, suffix = pattern.split("*")
    ngrams = []

    if prefix:
        # wildcard index has n-grams max length of 3 and $ occupies 1
        # will generate $le
        ngrams.append("$" + prefix[:2])

    if suffix:
        # will generate "ng$"
        ngrams.append(suffix[-2:] + "$")

    candidate_terms = None

    for ngram in ngrams:
        # use n_gram index built earlier from builders.py accessed through access.py
        matches = set(find_wildcard_matches(ngram, index_path))

        if candidate_terms is None:
            candidate_terms = matches
        else: candidate_terms &= matches

    # this function's job is to return a list of terms that might match
    # example candidates = ["learning", "leaning", "leaping", "leading"]
    return sorted(candidate_terms) if candidate_terms is not None else []


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

    # this function's job is to check if previous function's candidate terms start and end correctly.
    # e.g. learning.startswith("learn") ✅ and learning.endswith("ing") ✅    -> true
    # e.g. leaning.startswith("learn") ❌  and leaning.endswith("ing") ✅     -> false
    # e.g. leaping.startswith("learn") ❌  and leaping.endswith("ing") ✅     -> false
    # e.g. leading.startswith("learn") ❌  and leading.endswith("ing") ✅     -> false
    prefix, suffix = pattern.split("*")

    matching_terms = []
    for term in candidates:
        if term.startswith(prefix) and term.endswith(suffix) and len(term) >= len(prefix) + len(suffix):
            matching_terms.append(term)

    return sorted(matching_terms)

def _matching_terms_to_doc_ids(
    matched_terms: List[str],
    index_path: str,
) -> Set[int]:
    """Return documents containing any matched vocabulary term."""
    # This is the matched terms -> posting lists -> document IDs stage in the
    # Week 04 lecture section "Wildcard Expansion Pipeline".
    # TODO(Task 1): combine the posting lists.

    # e.g. learning --> {10, 30}
    doc_ids: Set[int] = set()

    for term in matched_terms:
        doc_ids.update(get_posting_list(term, index_path))

    # returns a set containing {10, 30}
    return doc_ids


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
