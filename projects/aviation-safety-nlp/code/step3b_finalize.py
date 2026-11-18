"""
STEP 3b: lock in the best clustering setting and build review-ready outputs.

Why: in step 3 the three finalists tied on rank (coherence was missing because gensim
is not installed), and the tie went to the LEAST stable one (stability ARI 0.60).
This script re-runs the most stable finalist (ARI 0.905) and adds:
  - cleaner keywords (generic words like "aircraft"/"flight" removed)
  - coherence computed WITHOUT gensim (document-level NPMI)
  - soft assignment of unclustered reports (kept in a separate column)
  - a list of near-duplicate clusters that could be merged

Input : asrs_clean.csv, emb_full_mpnet.npy   (both already on your computer)
Output: asrs_with_clusters_v2.csv, topics_v2.csv, similar_clusters.csv, cluster_map_v2.png

Run: python step3b_finalize.py        (about 3-5 minutes)
"""

import itertools
import warnings

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

SEED = 42
CFG = dict(n_neighbors=15, n_components=5, min_cluster_size=30, min_samples=10, method="eom")
SOFT_MIN = 0.20   # unclustered report joins a cluster only if its membership score >= this

JUNK = {"redacted", "xx", "xy", "xa", "xxl", "xxr", "xxx", "zz", "zzz", "abcd", "nan"}
GENERIC = {"aircraft", "flight", "air", "control", "captain", "officer", "plane", "crew",
           "pilot", "flying", "requested", "called", "issued", "left", "right"}
STOP = set(CountVectorizer(stop_words="english").get_stop_words()) | JUNK | GENERIC
TOKEN = r"(?u)\b[a-z]{3,}\b"

clean = pd.read_csv("asrs_clean.csv", dtype=str).fillna("")
texts = clean["text"].str.replace(r"\s*;\s*", ", ", regex=True).tolist()
n = len(clean)
R = np.load("emb_full_mpnet.npy")


def reduce(X, nc, seed=SEED):
    return umap.UMAP(n_neighbors=CFG["n_neighbors"], n_components=nc, min_dist=0.0,
                     metric="cosine", random_state=seed).fit_transform(X)


def fit(U, predict=False):
    return hdbscan.HDBSCAN(min_cluster_size=CFG["min_cluster_size"], min_samples=CFG["min_samples"],
                           cluster_selection_method=CFG["method"], prediction_data=predict).fit(U)


def keywords(labels, topn=10):
    df = pd.DataFrame({"t": texts, "c": labels})
    df = df[df["c"] >= 0]
    docs = df.groupby("c")["t"].apply(" ".join)
    cv = CountVectorizer(stop_words=list(STOP), token_pattern=TOKEN, min_df=2)
    X = cv.fit_transform(docs.values).toarray().astype(float)
    ctf = (X / X.sum(1, keepdims=True)) * np.log(1 + X.sum(1).mean() / X.sum(0))
    vocab = np.array(cv.get_feature_names_out())
    return {c: vocab[np.argsort(-ctf[i])[:topn]].tolist() for i, c in enumerate(docs.index)}


def npmi_coherence(topics):
    """Document-level NPMI of each cluster's top words (range -1..1, higher = more coherent)."""
    cv = CountVectorizer(stop_words=list(STOP), token_pattern=TOKEN, binary=True, min_df=2)
    X = cv.fit_transform(texts).tocsc()
    voc, N = cv.vocabulary_, X.shape[0]
    dfreq = np.asarray(X.sum(0)).ravel()
    per_topic = []
    for t in topics:
        idx = [voc[w] for w in t if w in voc]
        vals = []
        for a, b in itertools.combinations(idx, 2):
            co = X[:, a].multiply(X[:, b]).sum()
            if co == 0:
                vals.append(-1.0)
            elif co < N:
                pij, pi, pj = co / N, dfreq[a] / N, dfreq[b] / N
                vals.append(np.log(pij / (pi * pj)) / -np.log(pij))
        per_topic.append(np.mean(vals) if vals else np.nan)
    return float(np.nanmean(per_topic)), per_topic


# ---------------------------------------------------------------- final clustering
U = reduce(R, CFG["n_components"])
h = fit(U, predict=True)
lab, prob = h.labels_, h.probabilities_
k = int(lab.max() + 1)

mem = hdbscan.all_points_membership_vectors(h)
best_mem = mem.max(axis=1)
soft = np.where(lab >= 0, lab, np.where(best_mem >= SOFT_MIN, mem.argmax(axis=1), -1))

xy = reduce(R, 2)
kw = keywords(lab)
order = sorted(kw)
coh, per_topic = npmi_coherence([kw[c] for c in order])
div = len({w for c in order for w in kw[c]}) / (10 * len(order))

# stability: 3 random 80% subsamples vs the full clustering
aris = []
for s in range(3):
    sel = np.sort(np.random.default_rng(100 + s).choice(n, int(0.8 * n), replace=False))
    sub_lab = fit(reduce(R[sel], CFG["n_components"], seed=100 + s)).labels_
    ok = (lab[sel] >= 0) & (sub_lab >= 0)
    aris.append(adjusted_rand_score(lab[sel][ok], sub_lab[ok]))

# ---------------------------------------------------------------- outputs
keep = [c for c in ["acn", "date", "flight_phase", "primary_problem", "anomaly", "synopsis"] if c in clean.columns]
out = clean[["report_id"] + keep].copy()
out["cluster"], out["cluster_prob"] = lab, prob.round(3)
out["cluster_soft"], out["soft_membership"] = soft, best_mem.round(3)
out["umap_x"], out["umap_y"] = xy[:, 0].round(3), xy[:, 1].round(3)
out["text"] = texts
out.to_csv("asrs_with_clusters_v2.csv", index=False, encoding="utf-8-sig")

rows = []
for c, coh_c in zip(order, per_topic):
    m = out[out.cluster == c].sort_values("cluster_prob", ascending=False)
    vc = m.primary_problem.value_counts(normalize=True)
    rows.append(dict(cluster=c, size=len(m), share_pct=round(len(m) / n * 100, 1),
                     keywords=", ".join(kw[c]), npmi=round(coh_c, 3),
                     top_primary_problem=vc.index[0], problem_purity_pct=round(vc.iloc[0] * 100),
                     top_flight_phase=m.flight_phase.str.split(";").str[0].value_counts().index[0],
                     example_synopses=" || ".join(s[:200] for s in m.synopsis.head(3))))
topics = pd.DataFrame(rows).sort_values("size", ascending=False)
topics.to_csv("topics_v2.csv", index=False, encoding="utf-8-sig")

cent = np.vstack([R[lab == c].mean(0) for c in order])
cent /= np.linalg.norm(cent, axis=1, keepdims=True)
sim = cent @ cent.T
pairs = [(order[i], order[j], sim[i, j]) for i in range(len(order)) for j in range(i + 1, len(order))]
pairs = sorted(pairs, key=lambda p: -p[2])[:15]
pd.DataFrame([dict(cluster_a=a, cluster_b=b, centroid_cosine=round(s, 3),
                   keywords_a=", ".join(kw[a][:6]), keywords_b=", ".join(kw[b][:6]))
              for a, b, s in pairs]).to_csv("similar_clusters.csv", index=False)

plt.figure(figsize=(9, 7))
noise = lab < 0
plt.scatter(xy[noise, 0], xy[noise, 1], s=3, c="lightgrey")
plt.scatter(xy[~noise, 0], xy[~noise, 1], s=3, c=lab[~noise], cmap="nipy_spectral")
plt.title(f"ASRS clusters v2 ({k} clusters, {noise.mean() * 100:.0f}% unclustered)")
plt.savefig("cluster_map_v2.png", dpi=150, bbox_inches="tight")

print(f"\nSetting: {CFG}")
print(f"{k} clusters | unclustered {noise.mean() * 100:.1f}% -> {(soft < 0).mean() * 100:.1f}% after soft assignment (SOFT_MIN={SOFT_MIN})")
print(f"Coherence (doc-level NPMI): {coh:.3f} | topic diversity: {div:.3f} | stability ARI (3 runs): {np.mean(aris):.3f}")
print(f"Largest cluster: {np.bincount(lab[lab >= 0]).max() / n * 100:.1f}% of reports")
print(topics[["cluster", "size", "keywords", "npmi"]].head(12).to_string(index=False))
print("\nSaved: asrs_with_clusters_v2.csv, topics_v2.csv, similar_clusters.csv, cluster_map_v2.png")
