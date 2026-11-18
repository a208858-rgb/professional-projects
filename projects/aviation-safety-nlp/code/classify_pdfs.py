"""
Classify Malaysian aviation safety report PDFs (AAIB-style) into ONE growing CSV.

Each run: reads new PDFs from a folder, extracts the narrative + key facts, predicts
theme / novelty / primary problem / flight phase / anomaly, and APPENDS to the CSV.
Files already processed (same content) are skipped, so you can run it any time.

Usage:
    python classify_pdfs.py                              # uses ./reports and writes malaysia_reports_classified.csv
    python classify_pdfs.py --pdf-dir reports --out malaysia_reports_classified.csv
    python classify_pdfs.py --force                      # re-run everything after you retrain the model
                                                         # (your review_* columns are kept)
Needs: pip install pymupdf sentence-transformers scikit-learn joblib pandas numpy
"""

import argparse
import datetime as dt
import hashlib
import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from asrs_text import chunk, clean

REVIEW_COLS = ["review_theme", "review_primary_problem", "review_flight_phase", "review_anomaly", "reviewer"]
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
MALAY_WORDS = {"yang", "dan", "di", "ini", "itu", "dengan", "untuk", "pada", "dari", "tidak", "ke", "kepada",
               "adalah", "telah", "kapal", "terbang", "juruterbang", "lapangan"}

# ------------------------------------------------------------------ PDF reading
def read_pdf(path):
    try:
        try:
            import pymupdf as fitz      # PyMuPDF (new name)
        except ImportError:
            import fitz                 # PyMuPDF (old name)
        with fitz.open(path) as doc:
            return [p.get_text("text") for p in doc]
    except ImportError:
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(path).pages]


def strip_boilerplate(pages):
    """Remove running headers/footers and bare page numbers."""
    norm = lambda l: re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip()))
    counts = Counter(norm(l) for p in pages for l in set(p.splitlines()) if l.strip())
    limit = max(3, int(0.4 * len(pages)))
    out = []
    for p in pages:
        keep = []
        for l in p.splitlines():
            s = l.strip()
            if (counts[norm(l)] >= limit and len(s) < 80) or re.fullmatch(r"[ivxlcdm]{1,6}|\d{1,3}|[A-H]-?\d{1,2}", s, re.I):
                continue
            keep.append(l)
        out.append("\n".join(keep))
    return "\n".join(out)


# ------------------------------------------------------------------ sections
H_SYNOPSIS = re.compile(r"^\s*SYNOPSIS\s*$", re.M | re.I)
H_HISTORY = re.compile(r"^\s*1\.1\s+History of the Flight\s*$", re.M | re.I)
H_CAUSE = re.compile(r"^\s*3\.2\s+Cause\s*/?\s*Contributing Factors?\s*$", re.M | re.I)
H_ANY = re.compile(r"^\s*(?:\d\.\d{1,2}\s+[A-Z][^\n]{3,70}|\d\.0\s+[A-Z][A-Z ,/&-]{3,}|SYNOPSIS|CONCLUDING STATEMENT)\s*$", re.M)


def section(text, pattern):
    """Longest block starting at a heading and ending at the next heading
    (so a table-of-contents entry, which is followed immediately by another heading, loses)."""
    starts = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if not starts:
        return ""
    ends = sorted(m.start() for m in H_ANY.finditer(text))
    best = ""
    for s, e in starts:
        nxt = next((x for x in ends if x > e), len(text))
        body = text[e:nxt].strip()
        if len(body) > len(best):
            best = body
    return best


# ------------------------------------------------------------------ metadata
def field(text, label):
    m = re.search(rf"^\s*{label}\s*:\s*(.+)$", text, re.M | re.I)
    return m.group(1).strip() if m else ""


def parse_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    return ""


def metadata(full, first_pages, cover):
    head = first_pages
    low = cover.lower()
    rtype = next((t for t, k in [("final", "final report"), ("interim", "interim"),
                                 ("preliminary", "preliminary")] if k in low), "")
    eclass = next((c for c, k in [("serious incident", "serious incident"), ("accident", "accident"),
                                  ("incident", "incident")] if k in low), "")
    no = re.search(r"REPORT\s+NO\.?[ \t]*:?[ \t]*([A-Z]{1,2}[ \t]?\d{1,3}[ \t]?/[ \t]?\d{2}[A-Z]?)", head, re.I) \
        or re.search(r"\b([AIS][ \t]\d{2}/\d{2}[A-Z]?)\b", head)
    dt_field = field(head, r"DATE(?:\s+AND\s+TIME)?")
    date = parse_date(dt_field) or parse_date(re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", head).group(1)
                                              if re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", head) else "")
    tm = re.search(r"(\d{4})\s*(?:LT|hrs)", dt_field)
    reg = field(head, "REGISTRATION") or (re.search(r"\b(9M-[A-Z]{3})\b", head) or [None, ""])[1]
    cat = re.search(r"coded as (?:an? )?([^()\n]{3,80})\(([A-Z][A-Z0-9-]{1,8})\)", full)
    fatal = re.search(r"Fatal\s+(\d+|Nil)\s+(\d+|Nil)\s+(\d+|Nil)\s+(\d+|Nil)", full)
    fix = lambda v: re.sub(r"(?<=[A-Za-z])\d{1,2}$", "", v).strip()   # drop footnote digit
    return dict(
        report_type=rtype, event_class=eclass, report_no=re.sub(r"\s+", " ", no.group(1)).strip() if no else "",
        report_date=date, report_time_lt=tm.group(1) if tm else "",
        operator=fix(field(head, "OPERATOR")), aircraft_type=field(head, "AIRCRAFT TYPE"),
        registration=reg, location=fix(field(head, r"PLACE OF OCCURRENCE")),
        fatal_total=(0 if fatal.group(4).lower() == "nil" else int(fatal.group(4))) if fatal else "",
        adrep_category=cat.group(2) if cat else "")


# ------------------------------------------------------------------ embedding + prediction
_embedder = None


def embed_texts(model_name, texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
    return _embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def predict(art, v):
    th, ref = art["theme"], art["reference"]
    P = th["clf"].predict_proba(v[None])[0]
    order = np.argsort(-P)[:3]
    cls = th["clf"].classes_
    maxsim = float((th["centroids"] @ v).max())
    top = int(cls[order[0]])
    status = "novel" if maxsim < th["novelty_sim_threshold"] else \
        ("low_confidence" if P[order[0]] < th["low_conf_threshold"] else "assigned")
    pp = art["problem"]["clf"].predict_proba(v[None])[0]
    pf = art["phase"]["clf"].predict_proba(v[None])[0]
    pa = art["anomaly"]["clf"].predict_proba(v[None])[0]
    pick = lambda p, labels, t: [labels[i] for i in np.where(p >= t)[0]] or [labels[int(p.argmax())]]
    sims = ref["vectors"].astype(np.float32) @ v
    nn = np.argsort(-sims)[:3]
    return dict(
        theme_id=top, theme=th["names"].get(top, str(top)), theme_conf=round(float(P[order[0]]), 3),
        theme_top3=" | ".join(f"{th['names'].get(int(cls[i]), cls[i])} ({P[i]:.2f})" for i in order),
        theme_status=status, novelty_sim=round(maxsim, 3),
        primary_problem=art["problem"]["clf"].classes_[int(pp.argmax())], primary_problem_conf=round(float(pp.max()), 3),
        flight_phase="; ".join(pick(pf, art["phase"]["labels"], art["phase"]["threshold"])),
        anomaly="; ".join(pick(pa, art["anomaly"]["labels"], art["anomaly"]["threshold"])),
        nearest_asrs_acn=" | ".join(str(ref["acn"][i]) for i in nn),
        nearest_asrs_sim=" | ".join(f"{sims[i]:.2f}" for i in nn),
        nearest_asrs_synopsis_1=ref["synopsis"][nn[0]])


# ------------------------------------------------------------------ one PDF -> one row
def analyse(path, art):
    sha = hashlib.sha1(open(path, "rb").read()).hexdigest()
    row = dict(file_name=os.path.basename(path), file_sha1=sha)
    reasons = []
    pages = read_pdf(path)
    if sum(len(p) for p in pages) < 200 * max(1, len(pages)) * 0.25:
        reasons.append("no_text_layer_needs_OCR")
        row.update(needs_review=True, review_reasons="; ".join(reasons))
        return row
    full = strip_boilerplate(pages)
    first = "\n".join(pages[:3])
    row.update(metadata(full, first, pages[0]))

    syn, hist = section(full, H_SYNOPSIS), section(full, H_HISTORY)
    narrative, basis = (syn + "\n" + hist).strip(), "synopsis+history"
    if len(narrative.split()) < 80:
        body = re.sub(r"\s+", " ", full)
        narrative, basis = " ".join(body.split()[300:1100]), "fallback_words_300_to_1100"
        reasons.append("narrative_section_not_found")
    elif not hist:
        basis = "synopsis_only"
    cause = section(full, H_CAUSE)
    row.update(aaib_cause_text=re.sub(r"\s+", " ", cause)[:500], text_basis=basis)

    text = clean(narrative)
    words = text.split()
    chunks = chunk(text)
    row.update(n_words_used=len(words), n_chunks=len(chunks))
    malay = sum(w in MALAY_WORDS for w in words) / max(1, len(words))
    row["language_flag"] = "possible_malay" if malay > 0.03 else ""
    if malay > 0.03:
        reasons.append("possible_malay_text")

    E = embed_texts(art["embed_model"], chunks)
    v = E.mean(0)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    row.update(predict(art, v))

    if row["theme_status"] != "assigned":
        reasons.append(f"theme_{row['theme_status']}")
    for k in ["report_date", "registration"]:
        if not row.get(k):
            reasons.append(f"missing_{k}")
    row.update(needs_review=bool(reasons), review_reasons="; ".join(reasons))
    return row


# ------------------------------------------------------------------ main
COLS = ["file_name", "file_sha1", "report_type", "event_class", "report_no", "report_date", "report_time_lt",
        "operator", "aircraft_type", "registration", "location", "fatal_total", "adrep_category", "aaib_cause_text",
        "text_basis", "n_words_used", "n_chunks", "theme_id", "theme", "theme_conf", "theme_status", "theme_top3",
        "novelty_sim", "primary_problem", "primary_problem_conf", "flight_phase", "anomaly", "nearest_asrs_acn",
        "nearest_asrs_sim", "nearest_asrs_synopsis_1", "language_flag", "needs_review", "review_reasons",
        *REVIEW_COLS, "processed_at", "model_version"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="reports")
    ap.add_argument("--out", default="malaysia_reports_classified.csv")
    ap.add_argument("--model", default="models/asrs_theme_model_v1.joblib")
    ap.add_argument("--force", action="store_true", help="re-process all PDFs (keeps your review_* columns)")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.pdf_dir):
        raise SystemExit(f"PDF folder '{a.pdf_dir}' not found - create it and put the report PDFs inside.")
    art = joblib.load(a.model)
    old = pd.read_csv(a.out, dtype=str).fillna("") if os.path.exists(a.out) else pd.DataFrame(columns=COLS)
    done = set(old.file_sha1) if not a.force else set()
    pdfs = sorted(f for f in os.listdir(a.pdf_dir) if f.lower().endswith(".pdf"))
    new_rows = []
    for f in pdfs:
        path = os.path.join(a.pdf_dir, f)
        sha = hashlib.sha1(open(path, "rb").read()).hexdigest()
        if sha in done:
            print(f"skip (already done): {f}")
            continue
        try:
            row = analyse(path, art)
        except Exception as e:                         # one bad PDF must not stop the batch
            row = dict(file_name=f, file_sha1=sha, needs_review=True, review_reasons=f"error: {e}")
        row.update(processed_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), model_version=art["version"])
        new_rows.append(row)
        print(f"done: {f} -> {row.get('theme', '-')} [{row.get('theme_status', '-')}] review={row['needs_review']}")

    new = pd.DataFrame(new_rows)
    if len(new):
        if a.force and len(old):                       # carry over manual review columns
            keep = old.set_index("file_sha1")[REVIEW_COLS]
            for c in REVIEW_COLS:
                new[c] = new["file_sha1"].map(keep[c]).fillna("")
            old = old[~old.file_sha1.isin(new.file_sha1)]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = old
    for c in COLS:
        if c not in out:
            out[c] = ""
    out = out[COLS + [c for c in out.columns if c not in COLS]]
    for c in ["fatal_total", "n_words_used", "n_chunks", "theme_id"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"\n{len(new)} new row(s); {len(out)} total -> {a.out}")
    return out


if __name__ == "__main__":
    main()
