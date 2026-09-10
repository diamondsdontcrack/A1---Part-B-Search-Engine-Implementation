"""Optional preprocessing utility for Task 3 system experiments."""

from typing import List

from nltk.tokenize import word_tokenize

__all__ = ["preprocess"]


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Clean and tokenize a batch of noisy documents.
    Args:
        raw_html_list: One raw HTML or noisy web-text string per document.
    Returns:
        One token list per input document, preserving input order and length.
        Produce the final tokens with nltk.word_tokenize after cleaning. Return
        an empty list for a document with no usable text.
    TODO:
      Replace this TODO section with your own brief preprocessing procedure.
      Explain how your submitted implementation turns the provided noisy
      documents into cleaner text before tokenization. The explanation should
      be specific to choices you made after inspecting the dev data, not a
      generic HTML-cleaning description, and it must match your code.
    """
    pass
