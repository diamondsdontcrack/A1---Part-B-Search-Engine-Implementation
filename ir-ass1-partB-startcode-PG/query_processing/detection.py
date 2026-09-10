"""Query type detection for Task 1."""

import re


def detect_query_type(query: str) -> str:
    """Return ``proximity``, ``wildcard``, ``boolean``, or ``natural_language``.

    Detection is intentionally syntax-based:
    - ``NEAR/k`` -> proximity
    - ``*`` -> wildcard
    - uppercase ``AND``/``OR``/``NOT`` -> boolean
    - otherwise -> natural language

    Operators are case-sensitive in the assignment data.
    """
    if re.search(r"\bNEAR/\d+\b", query):
        return "proximity"
    if "*" in query:
        return "wildcard"
    if re.search(r"\b(AND|OR|NOT)\b", query):
        return "boolean"
    return "natural_language"
