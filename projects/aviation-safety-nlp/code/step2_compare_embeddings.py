"""
STEP 2 (v2): compare embedding models on a random sample of ASRS reports.

Input : asrs_clean.csv, asrs_chunks.csv  (from step 1)
Output: results_embedding_comparison.csv
        emb_<model>.npy  (one vector per sampled report, for step 3)
        sample_ids.csv   (which reports were sampled, same row order as the .npy files)

Run: python step2_compare_embeddings.py
Needs: pip install sentence-transformers umap-learn hdbscan scikit-learn
"""

import time

import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

SAMPLE = 2000          # reports to test on (set to None to use all, much slower)
SEED = 42
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # fast
    "bge_small": "BAAI/bge-small-en-v1.5",                # fast, strong
    "mpnet": "sentence-transformers/all-mpnet-base-v2",   # slower, often best
}

clean = pd.read_csv("asrs_clean.csv", dtype=str).fillna("")
chunks = pd.read_csv("asrs_chunks.csv", dtype=str).fillna("")

# The ASRS export turns commas into semicolons; restore them so the text reads naturally.
chunks["text"] = chunks["text"].str.replace(r"\s*;\s*", ", ", regex=True)

# --- sample reports
if SAMPLE and SAMPLE < len(clean):
    clean = clean.sample(SAMPLE, random_state=SEED)
ids = clean["report_id"].tolist()
sub = chunks[chunks["report_id"].isin(ids)].reset_index(drop=True)
print(f"Testing on {len(ids)} reports ({len(sub)} chunks)")
pd.Series(ids, name="report_id").to_csv("sample_ids.csv", index=False)

# --- evaluation labels (used ONLY for scoring, never for training).
# Note: ASRS lists anomalies alphabetically, so "first value" is NOT the main anomaly.
# We therefore use (a) the single-valued primary problem, (b) flight phase, and
# (c) the SET of anomaly categories, scored with neighbour overlap (Jaccard).
ANOM_PREFIXES = [
    ("Aircraft Equipment Problem", "Aircraft Equipment Problem"),
    ("ATC Issue", "ATC Issue"),
    ("Airspace Violation", "Airspace Violation"),
    ("Conflict", "Conflict"),
    ("Deviation / Discrepancy", "Procedural Deviation"),
    ("Deviation - ", "Flight Path Deviation"),
    ("Flight Deck / Cabin", "Flight Deck / Cabin Event"),
    ("Ground Event", "Ground Event"),
    ("Ground Excursion", "Ground Excursion"),
    ("Ground Incursion", "Ground Incursion"),
    ("Inflight Event", "Inflight Event"),
    ("No Specific Anomaly", "No Specific Anomaly"),
]


def anomaly_categories(cell):
    cats = set()
    for part in cell.split(";"):
        p = part.strip()
        if not p or p in ("Critical", "Less Severe"):   # severity tags, not anomalies
            continue
        for prefix, name in ANOM_PREFIXES:
            if p.startswith(prefix):
                cats.add(name)
                break
        else:
            cats.add(p)
    return cats


y_problem = clean["primary_problem"].str.strip().to_numpy(dtype=object)
y_phase = clean["flight_phase"].str.split(";").str[0].str.strip().to_numpy(dtype=object)
anom_sets = [anomaly_categories(c) for c in clean["anomaly"]]
print("Labels: primary_problem, flight_phase (first value), anomaly categories (set)")


def neighbours(R, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(R)
    return nn.kneighbors(R, return_distance=False)[:, 1:]


def agree_single(idx, y):
    """Share of neighbours with the same label, vs. chance (labelled reports only)."""
    labelled = y != ""
    ok = labelled[:, None] & labelled[idx]
    agree = ((y[idx] == y[:, None]) & ok).sum() / ok.sum()
    p = pd.Series(y[labelled]).value_counts(normalize=True)
    return agree, float((p ** 2).sum())


def agree_sets(idx, sets, seed=SEED):
    """Mean Jaccard overlap of anomaly sets between neighbours, vs. random pairs."""
    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else np.nan
    near = [jac(sets[i], sets[j]) for i in range(len(sets)) for j in idx[i]]
    rng = np.random.default_rng(seed)
    rand = [jac(sets[i], sets[j]) for i, j in
            zip(rng.integers(0, len(sets), 20000), rng.integers(0, len(sets), 20000))]
    return float(np.nanmean(near)), float(np.nanmean(rand))


results = []
for short, name in MODELS.items():
    print(f"\n=== {short} ({name}) ===")
    t0 = time.time()
    model = SentenceTransformer(name)
    E = model.encode(sub["text"].tolist(), batch_size=64,
                     normalize_embeddings=True, show_progress_bar=True)

    # average chunk vectors back to one vector per report
    R = (pd.DataFrame(E).assign(report_id=sub["report_id"].values)
         .groupby("report_id").mean().reindex(ids).values)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    np.save(f"emb_{short}.npy", R)
    secs = time.time() - t0

    row = {"model": short, "dims": R.shape[1], "minutes": round(secs / 60, 1)}

    # 1) neighbour agreement with expert labels (higher than chance = better space)
    idx = neighbours(R)
    for key, y in [("problem", y_problem), ("phase", y_phase)]:
        a, base = agree_single(idx, y)
        row[f"knn_{key}"], row[f"knn_{key}_chance"] = round(a, 3), round(base, 3)
    a, base = agree_sets(idx, anom_sets)
    row["anomaly_overlap"], row["anomaly_overlap_chance"] = round(a, 3), round(base, 3)

    # 2) quick clustering check
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                  random_state=SEED).fit_transform(R)
    cl = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(U)
    row["n_clusters"] = int(cl.max() + 1)
    row["outlier_pct"] = round((cl == -1).mean() * 100, 1)
    ok = (cl != -1) & (y_problem != "")
    row["nmi_vs_problem"] = round(normalized_mutual_info_score(y_problem[ok], cl[ok]), 3)

    print(row)
    results.append(row)

res = pd.DataFrame(results)
res.to_csv("results_embedding_comparison.csv", index=False)
print("\n", res.to_string(index=False))
print("\nSaved results_embedding_comparison.csv, sample_ids.csv and emb_*.npy")
