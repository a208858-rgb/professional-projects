"""
STEP 1: prepare ASRS narratives for embedding + clustering.

Input : asrs_combined.csv
Output: asrs_clean.csv   (one row per report, cleaned text)
        asrs_chunks.csv  (long narratives split into ~200-word chunks)

Run: python step1_prepare_text.py
"""

import re

import pandas as pd

INPUT = "asrs_combined.csv"
MIN_WORDS = 20          # drop reports shorter than this
CHUNK_WORDS = 200       # ~256 tokens, the limit of MiniLM-type embedding models

# Uppercase ASRS shorthand -> plain words (case-sensitive, so normal words are untouched)
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
# ASRS hides names/places as ZZZ, ZZZ1, XXX ... -> neutral token
REDACT_RE = re.compile(r"\b(?:Z{2,}\d*|z{2,}\d*|X{3,}|x{3,})\b")


def find_col(df, *keys):
    for c in df.columns:
        if all(k in c for k in keys):
            return c
    return None


def clean(text):
    t = str(text)
    t = ABBREV_RE.sub(lambda m: ABBREV[m.group(1)], t)
    t = REDACT_RE.sub("redacted", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
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


df = pd.read_csv(INPUT, dtype=str).fillna("")
print("Rows loaded:", len(df))
print("\nAll columns:")
for i, c in enumerate(df.columns):
    print(f"{i:3d}  {c}")

narr = find_col(df, "narrative")
syn = find_col(df, "synopsis")
acn = find_col(df, "acn")
if not narr:
    raise SystemExit("\nNo narrative column found. Send me the column list above.")
print(f"\nUsing narrative column: {narr} | synopsis: {syn} | id: {acn}")

df["report_id"] = df[acn] if acn else df.index.astype(str)
df["text"] = df[narr].map(clean)
if syn:
    df["synopsis_clean"] = df[syn].map(clean)

# --- quality filters
n0 = len(df)
df["n_words"] = df["text"].str.split().str.len()
df = df[df["n_words"] >= MIN_WORDS]
n1 = len(df)
df = df.drop_duplicates(subset="text")
n2 = len(df)
df = df.reset_index(drop=True)
print(f"\nDropped {n0 - n1} too-short reports and {n1 - n2} duplicate narratives -> {n2} kept")

# --- length stats
print("\nNarrative length (words):")
print(df["n_words"].describe(percentiles=[.5, .9, .95]).round(1))
print(f"Share longer than {CHUNK_WORDS} words: {(df['n_words'] > CHUNK_WORDS).mean():.1%}")

# --- chunks (one row per chunk, linked back by report_id)
rows = []
for rid, t in zip(df["report_id"], df["text"]):
    for j, c in enumerate(chunk(t)):
        rows.append({"report_id": rid, "chunk_no": j, "text": c})
chunks = pd.DataFrame(rows)
print(f"\nChunks: {len(chunks)} (from {len(df)} reports)")

# --- save
df.to_csv("asrs_clean.csv", index=False, encoding="utf-8-sig")
chunks.to_csv("asrs_chunks.csv", index=False, encoding="utf-8-sig")

print("\nTwo cleaned examples:")
for t in df["text"].sample(2, random_state=1):
    print("-", t[:400], "...\n")
print("Saved asrs_clean.csv and asrs_chunks.csv")
