"""Transcript normalisation shared by every ASR script.

The ASR alphabet is lower-case Turkish letters, apostrophe and space. Training,
quality filtering and evaluation must normalise identically, otherwise labels
and references drift apart, so this lives in one place.
"""
from __future__ import annotations

import re

# str.lower() follows the locale-free Unicode rules: "I" -> "i" and
# "İ" -> "i" + U+0307. Turkish needs "I" -> "ı" and "İ" -> "i", so map those
# two letters before lowering ("Işık" must become "ışık", not "işik").
_TURKISH_CASE = str.maketrans({"I": "ı", "İ": "i"})


def normalise(text: str) -> str:
    text = text.translate(_TURKISH_CASE).lower().replace("i̇", "i")
    text = re.sub(r"[^a-zçğıöşü' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
