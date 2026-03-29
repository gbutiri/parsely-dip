"""Sentence splitting for multi-sentence input.

Reserved for future expansion. Currently passes input through unchanged.
"""


def split_sentences(text):
    """Split text into sentences.

    For now, returns the input as a single-item list.
    Future: integrate NLTK sent_tokenize or similar.

    Args:
        text: raw user input

    Returns:
        list of sentence strings
    """
    return [text.strip()] if text and text.strip() else []
