"""
Shared text helpers. MUST be the same code at training time and at prediction time,
otherwise new reports are processed differently from the ASRS reports the model learned from.
"""

import re

CHUNK_WORDS = 200  # ~256 tokens, the limit of MiniLM/MPNet-style embedding models

# Uppercase shorthand -> plain words (case-sensitive, so normal words are untouched)
ABBREV = {
    "ATC": "air traffic control", "TWR": "tower", "APCH": "approach",
    "ACFT": "aircraft", "RWY": "runway", "TWY": "taxiway", "CAPT": "captain",
    "F/O": "first officer", "FO": "first officer", "RA": "resolution advisory",
    "TA": "traffic advisory", "MEL": "minimum equipment list",
    "PIC": "pilot in command", "FMS": "flight management system",
    "ILS": "instrument landing system", "VFR": "visual flight rules",
    "IFR": "instrument flight rules",
}
ABBREV_RE = re.compile(r"(?<![A-Za-z/])(" + "|".join(re.escape(k) for k in ABBREV) + r")(?![A-Za-z/])")
REDACT_RE = re.compile(r"\b(?:Z{2,}\d*|z{2,}\d*|X{3,}|x{3,})\b")


def clean(text):
    t = str(text)
    t = ABBREV_RE.sub(lambda m: ABBREV[m.group(1)], t)
    t = REDACT_RE.sub("redacted", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"\s*;\s*", ", ", t)   # ASRS exports turn commas into semicolons
    return t


def chunk(text, max_words=CHUNK_WORDS):
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks
