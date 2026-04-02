# -*- coding: utf-8 -*-
"""
Deterministic labels for letter-repetition tasks.

- Closed-set: countries are listed in the question; the winner is computed from
  that list only (no world-knowledge guess).
- Global: uses pycountry's English short names (ISO 3166-1 alpha-2 entries).
  Ties on max count break lexicographically by name for a stable answer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

import pycountry

# Matches "... name: A, B, C?" at end of closed-set questions
_CLOSED_SET_TAIL = re.compile(
    r"Among these countries[^:]*:\s*(.+?)\?\s*$", re.IGNORECASE
)


def max_single_letter_count(text: str) -> int:
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return 0
    return max(Counter(letters).values())


def parse_closed_set_country_names(question: str) -> Optional[List[str]]:
    m = _CLOSED_SET_TAIL.search(question.strip())
    if not m:
        return None
    part = m.group(1).strip()
    names = [x.strip() for x in part.split(",") if x.strip()]
    return names or None


def winner_among(names: List[str]) -> str:
    best_name: Optional[str] = None
    best_score = -1
    for n in names:
        s = max_single_letter_count(n)
        if best_name is None or s > best_score or (s == best_score and n < best_name):
            best_name = n
            best_score = s
    assert best_name is not None
    return best_name


def closed_set_expected_answer(question: str) -> str:
    names = parse_closed_set_country_names(question)
    if not names:
        raise ValueError(
            "Not a recognized closed-set question (expected 'Among these countries...: A, B, ...?')."
        )
    return winner_among(names)


def global_expected_answer_pycountry() -> Tuple[str, int]:
    """Return (country_name, max_single_letter_count) over pycountry English names."""
    best_name: Optional[str] = None
    best_score = -1
    for c in pycountry.countries:
        name = c.name
        s = max_single_letter_count(name)
        if best_name is None or s > best_score or (s == best_score and name < best_name):
            best_name = name
            best_score = s
    assert best_name is not None
    return best_name, best_score


def relabel_closed_set_row(input_text: str, current_target: str) -> Tuple[str, bool]:
    """
    Return (oracle_target, mismatch_flag) where mismatch means JSON disagreed with oracle.
    """
    oracle = closed_set_expected_answer(input_text)
    return oracle, oracle != current_target
