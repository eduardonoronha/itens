"""
Utility helpers for text normalization and small text-processing helpers.
These are pulled out from `arq.py` to improve readability and separation of concerns.

Exports:
- remover_acentos(txt): remove unicode combining accents
- normalize(txt): uppercase, remove accents, replace problematic tokens and non-alphanumerics, return stripped string
"""

import re
import unicodedata


def remover_acentos(txt):
    """Remove unicode combining marks from `txt` and return the de-accented string."""

    if txt is None:
        return ""

    txt = unicodedata.normalize("NFKD", str(txt))

    return "".join(
        c for c in txt
        if not unicodedata.combining(c)
    )


def normalize(txt):
    """Normalize text for consistent matching.

    Rules:
    - convert to str and uppercase
    - remove accents
    - replace some known OCR artifacts
    - normalize decimal commas to dots
    - keep only A-Z, 0-9 and a few punctuation chars
    - collapse whitespace
    """

    if txt is None:
        return ""

    txt = str(txt)

    txt = remover_acentos(txt)

    txt = txt.upper()

    txt = txt.replace("_X0000_", " ")
    txt = txt.replace("X0000", " ")
    txt = txt.replace("ACOO", "ACO")
    txt = txt.replace("INOXX", "INOX")

    txt = re.sub(r"(\d),(\d)", r"\1.\2", txt)

    txt = re.sub(r"[^A-Z0-9./%+\-_]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt)

    return txt.strip()
