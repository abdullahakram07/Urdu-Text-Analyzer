"""Urdu-aware text normalization that preserves the original Unicode script."""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[\w\u0600-\u06FF]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_urdu(text: str) -> str:
    """Normalize Unicode and noise while retaining Urdu letters, diacritics and punctuation.

    Applies NFC (canonical composition), replaces URLs and @mentions with stable tokens,
    removes tatweel and format/control characters, then collapses whitespace. It does not
    transliterate, stem, remove stop words, strip diacritics, or unify Urdu letter variants.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text)
    normalized = _URL.sub(" URLTOKEN ", normalized)
    normalized = _MENTION.sub(" USERTOKEN ", normalized)
    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if char == "ـ":  # Arabic tatweel
            continue
        if category in {"Cc", "Cf", "Cs"} and char not in "\n\r\t":
            continue
        chars.append(char)
    return _WHITESPACE.sub(" ", "".join(chars)).strip()
