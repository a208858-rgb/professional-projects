"""
STEP 3: embed ALL reports, tune UMAP + HDBSCAN, evaluate candidates, assign clusters.

Input : asrs_clean.csv, asrs_chunks.csv
Output: emb_full_mpnet.npy         report vectors (cached, so re-runs are fast)
        results_tuning.csv         every UMAP/HDBSCAN setting tried + metrics
        results_top_configs.csv    top candidates with stability + coherence
        asrs_with_clusters.csv     every report with its cluster, probability, 2D coordinates
        topics.csv                 per cluster: size, keywords, example reports, label breakdown
        cluster_map.png            2D map of the clusters

Run: python step3_cluster_tuning.py
Needs: pip install sentence-transformers umap-learn hdbscan scikit-learn gensim matplotlib
Time : about 10 min embedding + 10-20 min tuning on a normal laptop.
"""

import itertools
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

MODEL = "sentence-transformers/all-mpnet-base-v2"
TAG = "mpnet"
SEED = 42
GRID = dict(
    n_neighbors=[15, 30, 50],
    n_components=[5, 10],
    min_cluster_size=[15, 30, 50],
    min_samples=[5, 10],
    method=["eom", "leaf"],
)
# A candidate must look interpretable to be considered:
MIN_CLUSTERS, MAX_CLUSTERS = 10, 80
MAX_OUTLIER_PCT = 45
MAX_LARGEST_PCT = 25     # no single cluster may swallow more than this % of reports

# placeholder tokens from ASRS anonymisation must not appear as "themes"
JUNK = {"redacted", "xx", "xy", "xa", "xxl", "xxr", "xxx", "zz", "zzz", "abcd", "nan"}
STOP = set(CountVectorizer(stop_words="english").get_stop_words()) | JUNK

clean = pd.read_csv("asrs_clean.csv", dtype=str).fillna("")
ids = clean["report_id"].tolist()
texts = clean["text"].str.replace(r"\s*;\s*", ", ", regex=True).tolist()
n = len(ids)


# ------------------------------------------------------------------ A. embeddings
def embed_all():
    path = f"emb_full_{TAG}.npy"
    if os.path.exists(path):
        print("Loading cached", path)
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    chunks = pd.read_csv("asrs_chunks.csv", dtype=str).fillna("")
    chunks["text"] = chunks["text"].str.replace(r"\s*;\s*", ", ", regex=True)
    E = SentenceTransformer(MODEL).encode(chunks["text"].tolist(), batch_size=64,
                                          normalize_embeddings=True, show_progress_bar=True)
    R = (pd.DataFrame(E).assign(report_id=chunks["report_id"].values)
         .groupby("report_id").mean().reindex(ids).values)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    np.save(path, R)
    return R


# ------------------------------------------------------------------ helpers
def reduce(R, nn, nc, seed=SEED):
    return umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine",
                     random_state=seed).fit_transform(R)


def cluster(U, mcs, ms, method, tree=False):
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                           cluster_selection_method=method, gen_min_span_tree=tree).fit(U)


def keywords(labels, topn=10):
    """c-TF-IDF: words that are frequent in one cluster but rare in the others."""
    df = pd.DataFrame({"t": texts, "c": labels})
    df = df[df["c"] >= 0]
    docs = df.groupby("c")["t"].apply(" ".join)
    cv = CountVectorizer(stop_words=list(STOP), token_pattern=r"(?u)\b[a-z]{3,}\b", min_df=2)
    X = cv.fit_transform(docs.values).toarray().astype(float)
    ctf = (X / X.sum(1, keepdims=True)) * np.log(1 + X.sum(1).mean() / X.sum(0))
    vocab = np.array(cv.get_feature_names_out())
    return {c: vocab[np.argsort(-ctf[i])[:topn]].tolist() for i, c in enumerate(docs.index)}


def coherence_diversity(topics):
    div = len({w for t in topics for w in t}) / sum(len(t) for t in topics)
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
        tok = [[w for w in re.findall(r"[a-z]{3,}", t) if w not in STOP] for t in texts]
        d = Dictionary(tok)
        topics = [[w for w in t if w in d.token2id] for t in topics]
        cm = CoherenceModel(topics=topics, texts=tok, dictionary=d, coherence="c_npmi",
                            topn=10, processes=1)
        return float(cm.get_coherence()), div
    except Exception as e:
        print("  (coherence skipped:", e, ")")
        return np.nan, div


def stability(R, cfg, base, n_runs=3, frac=0.8):
    """Re-run on random 80% subsamples; ARI against the full-data clustering."""
    scores = []
    for s in range(n_runs):
        sel = np.sort(np.random.default_rng(100 + s).choice(n, int(frac * n), replace=False))
        U = reduce(R[sel], cfg["n_neighbors"], cfg["n_components"], seed=100 + s)
        lab = cluster(U, cfg["min_cluster_size"], cfg["min_samples"], cfg["method"]).labels_
        ok = (base[sel] >= 0) & (lab >= 0)
        scores.append(adjusted_rand_score(base[sel][ok], lab[ok]) if ok.sum() > 50 else np.nan)
    return float(np.nanmean(scores))


# ------------------------------------------------------------------ B. grid search
R = embed_all()
problem = clean["primary_problem"].str.strip().to_numpy(dtype=object)
rows, U_cache = [], {}
for nn, nc in itertools.product(GRID["n_neighbors"], GRID["n_components"]):
    U_cache[(nn, nc)] = U = reduce(R, nn, nc)
    for mcs, ms, meth in itertools.product(GRID["min_cluster_size"], GRID["min_samples"], GRID["method"]):
        h = cluster(U, mcs, ms, meth, tree=True)
        lab, k = h.labels_, int(h.labels_.max() + 1)
        try:
            rel = float(h.relative_validity_) if k >= 2 else np.nan
        except Exception:
            rel = np.nan
        ok = (lab >= 0) & (problem != "")
        rows.append(dict(n_neighbors=nn, n_components=nc, min_cluster_size=mcs, min_samples=ms,
                         method=meth, n_clusters=k, outlier_pct=round((lab < 0).mean() * 100, 1),
                         largest_pct=round(np.bincount(lab[lab >= 0]).max() / n * 100, 1) if k else np.nan,
                         rel_validity=round(rel, 4),
                         nmi_vs_problem=round(normalized_mutual_info_score(problem[ok], lab[ok]), 3) if k >= 2 else np.nan))
    print(f"UMAP nn={nn}, dims={nc} done")

tuning = pd.DataFrame(rows)
tuning.to_csv("results_tuning.csv", index=False)
ok = tuning[tuning.n_clusters.between(MIN_CLUSTERS, MAX_CLUSTERS)
            & (tuning.outlier_pct <= MAX_OUTLIER_PCT) & (tuning.largest_pct <= MAX_LARGEST_PCT)]
if ok.empty:
    print("No setting met all interpretability limits; relaxing to n_clusters >= 5.")
    ok = tuning[tuning.n_clusters >= 5]
top = ok.sort_values("rel_validity", ascending=False).head(3)
print("\nTop candidates by relative validity:\n", top.to_string(index=False))

# ------------------------------------------------------------------ C. evaluate candidates
cands = []
for _, cfg in top.iterrows():
    cfg = cfg.to_dict()
    U = U_cache[(cfg["n_neighbors"], cfg["n_components"])]
    lab = cluster(U, cfg["min_cluster_size"], cfg["min_samples"], cfg["method"]).labels_
    coh, div = coherence_diversity(list(keywords(lab).values()))
    cfg.update(coherence_npmi=round(coh, 3), topic_diversity=round(div, 3),
               stability_ari=round(stability(R, cfg, lab), 3))
    print({k: cfg[k] for k in ["n_neighbors", "n_components", "min_cluster_size", "min_samples",
                               "method", "n_clusters", "coherence_npmi", "topic_diversity", "stability_ari"]})
    cands.append(cfg)
cand = pd.DataFrame(cands)
metrics = ["rel_validity", "coherence_npmi", "stability_ari"]
cand["avg_rank"] = cand[metrics].rank(ascending=False).mean(axis=1)
cand = cand.sort_values("avg_rank")
cand.to_csv("results_top_configs.csv", index=False)
best = cand.iloc[0].to_dict()
print("\nChosen setting:", {k: best[k] for k in ["n_neighbors", "n_components", "min_cluster_size", "min_samples", "method"]})

# ------------------------------------------------------------------ D. final clustering + outputs
U = U_cache[(int(best["n_neighbors"]), int(best["n_components"]))]
h = cluster(U, int(best["min_cluster_size"]), int(best["min_samples"]), best["method"])
lab, prob = h.labels_, h.probabilities_
xy = reduce(R, int(best["n_neighbors"]), 2)

keep = [c for c in ["acn", "date", "flight_phase", "primary_problem", "anomaly", "synopsis"] if c in clean.columns]
out = clean[["report_id"] + keep].copy()
out["cluster"], out["cluster_prob"] = lab, prob.round(3)
out["umap_x"], out["umap_y"] = xy[:, 0].round(3), xy[:, 1].round(3)
out["text"] = texts
out.to_csv("asrs_with_clusters.csv", index=False, encoding="utf-8-sig")

kw = keywords(lab)
trow = []
for c in sorted(kw):
    m = out[out.cluster == c]
    trow.append(dict(cluster=c, size=len(m), share_pct=round(len(m) / n * 100, 1),
                     keywords=", ".join(kw[c]),
                     top_primary_problem=m.primary_problem.value_counts().index[0],
                     top_flight_phase=m.flight_phase.str.split(";").str[0].value_counts().index[0],
                     example_report_ids=", ".join(m.sort_values("cluster_prob", ascending=False)
                                                   .report_id.head(3))))
pd.DataFrame(trow).sort_values("size", ascending=False).to_csv("topics.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 7))
noise = lab < 0
plt.scatter(xy[noise, 0], xy[noise, 1], s=3, c="lightgrey")
plt.scatter(xy[~noise, 0], xy[~noise, 1], s=3, c=lab[~noise], cmap="nipy_spectral")
plt.title(f"ASRS report clusters ({lab.max() + 1} clusters, {noise.mean() * 100:.0f}% unclustered)")
plt.savefig("cluster_map.png", dpi=150, bbox_inches="tight")

print(f"\n{lab.max() + 1} clusters | {noise.mean() * 100:.1f}% unclustered")
print(pd.DataFrame(trow).sort_values("size", ascending=False)[["cluster", "size", "keywords"]].head(15).to_string(index=False))
print("\nSaved: asrs_with_clusters.csv, topics.csv, results_tuning.csv, results_top_configs.csv, cluster_map.png")
