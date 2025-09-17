# src/expand.py
"""
Query expansion utilities for semantic search.

This module enriches user queries with related terms, improving recall
in vector search. It currently supports:

- WordNet synonym expansion
- (Placeholder) future embedding-based expansion

Functions:
    tokenize_for_expansion: Basic tokenizer for extracting candidate tokens.
    wordnet_expansion: Expand tokens with WordNet synonyms.
    expand_query: High-level wrapper for applying expansions.
"""

from typing import List, Set
import re
from nltk.corpus import wordnet as wn


def tokenize_for_expansion(text: str) -> List[str]:
    """
    Simple tokenizer for expansion candidates.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Lowercased tokens (length >= 3).
    """
    if not text:
        return []
    # Extract alphanumeric words (min length 3)
    tokens = re.findall(r"[A-Za-z0-9']{3,}", text.lower())
    return tokens


def wordnet_expansion(query: str, max_syn_per_word: int = 3) -> List[str]:
    """
    Expand query terms with WordNet synonyms.

    Args:
        query (str): Original user query.
        max_syn_per_word (int): Maximum synonyms per token.

    Returns:
        List[str]: Synonym expansions (lowercased, unique).
    """
    tokens = tokenize_for_expansion(query)
    expansions: Set[str] = set()

    for t in tokens:
        synsets = wn.synsets(t)
        count = 0
        for s in synsets:
            for lemma in s.lemmas():
                w = lemma.name().replace("_", " ")
                # Avoid echoing the same word
                if w.lower() != t:
                    expansions.add(w.lower())
                    count += 1
                    if count >= max_syn_per_word:
                        break
            if count >= max_syn_per_word:
                break

    return list(expansions)


def expand_query(query: str, use_wordnet: bool = True) -> List[str]:
    """
    High-level query expansion.

    Args:
        query (str): User query.
        use_wordnet (bool): Whether to use WordNet synonyms.

    Returns:
        List[str]: Expanded terms (possibly empty).
    """
    extras: List[str] = []

    if use_wordnet:
        extras.extend(wordnet_expansion(query))

    # Placeholder: other expansion strategies can be added here
    # e.g., co-occurrence stats, embedding similarity, or custom vocab

    return extras
