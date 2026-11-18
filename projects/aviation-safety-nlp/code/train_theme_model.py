"""
STEP 4a: train the model that turns a report's text vector into the output columns:
  theme (your 38 clusters) + novelty flag, primary problem, flight phase(s), anomaly categories.

Input : asrs_with_clusters_v2.csv   (reports + clusters, from step 3b)
        emb_full_mpnet.npy          (report vectors, from step 3)
        cluster_names.csv           (edit the names; keep the cluster ids)
Output: models/asrs_theme_model_v1.joblib   (the deployable model)
        models/metrics.json, models/model_card.md, models/leave_theme_out.csv

Run: python train_theme_model.py        (about 5-10 minutes)
Needs: pip install scikit-learn pandas numpy joblib
"""

import json
import os
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

CLUSTERS_CSV, EMB, NAMES = "asrs_with_clusters_v2.csv", "emb_full_mpnet.npy", "cluster_names.csv"
OUT_DIR, VERSION, SEED = "models", "asrs-themes-v1", 42
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
MIN_LABEL_COUNT = 30      # phases / anomaly categories rarer than this are not modelled
LOW_CONF = 0.35           # theme probability below this -> needs review
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CLUSTERS_CSV, dtype=str).fillna("")
df["cluster"] = df["cluster"].astype(int)
R = np.load(EMB).astype(np.float32)
assert len(R) == len(df), f"{len(R)} vectors but {len(df)} reports - files are out of sync"
n = len(df)
names = pd.read_csv(NAMES).set_index("cluster")["theme_name"].to_dict()

# ----------------------------------------------------------------- targets
ANOM_PREFIXES = [
    ("Aircraft Equipment Problem", "Aircraft Equipment Problem"), ("ATC Issue", "ATC Issue"),
    ("Airspace Violation", "Airspace Violation"), ("Conflict", "Conflict"),
    ("Deviation / Discrepancy", "Procedural Deviation"), ("Deviation - ", "Flight Path Deviation"),
    ("Flight Deck / Cabin", "Flight Deck / Cabin Event"), ("Ground Event", "Ground Event"),
    ("Ground Excursion", "Ground Excursion"), ("Ground Incursion", "Ground Incursion"),
    ("Inflight Event", "Inflight Event"), ("No Specific Anomaly", "No Specific Anomaly"),
]


def anomaly_categories(cell):
    cats = set()
    for part in cell.split(";"):
        p = part.strip()
        if not p or p in ("Critical", "Less Severe"):
            continue
        for prefix, name in ANOM_PREFIXES:
            if p.startswith(prefix):
                cats.add(name)
                break
        else:
            cats.add(p)
    return sorted(cats)


def keep_common(lists):
    cnt = Counter(x for l in lists for x in l)
    keep = sorted(x for x, c in cnt.items() if c >= MIN_LABEL_COUNT)
    return keep, [[x for x in l if x in keep] for l in lists]


phase_labels, phase_lists = keep_common([sorted({p.strip() for p in c.split(";") if p.strip()}) for c in df.flight_phase])
anom_labels, anom_lists = keep_common([anomaly_categories(c) for c in df.anomaly])

prob_counts = df.primary_problem.value_counts()
common_prob = set(prob_counts[prob_counts >= 20].index) - {""}
y_prob = df.primary_problem.where(df.primary_problem.isin(common_prob), "Other").to_numpy(dtype=object)
y_prob[df.primary_problem.values == ""] = ""

has_theme = (df.cluster >= 0).values
has_prob = y_prob != ""
has_phase = np.array([len(l) > 0 for l in phase_lists])
has_anom = np.array([len(l) > 0 for l in anom_lists])
print(f"{n} reports | in a theme: {has_theme.sum()} | phase labels: {phase_labels} | anomaly labels: {anom_labels}")

# ----------------------------------------------------------------- split (same for every target)
idx = np.arange(n)
train, test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=df.cluster)
metrics = {"n_reports": n, "n_train": len(train), "n_test": len(test)}


def sel(ix, mask):
    return ix[mask[ix]]


def pick_C(X, y, grid=(2, 8, 32)):
    scores = {C: cross_val_score(LogisticRegression(C=C, max_iter=3000), X, y, cv=3, scoring="f1_macro").mean() for C in grid}
    best = max(scores, key=scores.get)
    print(f"   C search {({k: round(v, 3) for k, v in scores.items()})} -> {best}")
    return best


def ece(P, y, classes, bins=10):
    conf, pred = P.max(1), classes[P.argmax(1)]
    right = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(right[m].mean() - conf[m].mean())
    return float(tot)


def single_label(name, y, mask):
    tr, te = sel(train, mask), sel(test, mask)
    print(f"\n[{name}] train {len(tr)} / test {len(te)}")
    C = pick_C(R[tr], y[tr])
    clf = LogisticRegression(C=C, max_iter=3000).fit(R[tr], y[tr])
    P = clf.predict_proba(R[te])
    pred = clf.classes_[P.argmax(1)]
    maj = pd.Series(y[tr]).value_counts().index[0]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english", max_features=50000)
    Xtr = tf.fit_transform(df.text.iloc[tr])
    base = LogisticRegression(C=10, max_iter=3000).fit(Xtr, y[tr])
    bpred = base.predict(tf.transform(df.text.iloc[te]))
    m = dict(C=C, accuracy=accuracy_score(y[te], pred), macro_f1=f1_score(y[te], pred, average="macro"),
             top3_accuracy=top_k_accuracy_score(y[te], P, k=min(3, len(clf.classes_)), labels=clf.classes_),
             ece=ece(P, y[te], clf.classes_),
             baseline_majority_accuracy=float((y[te] == maj).mean()),
             baseline_tfidf_accuracy=accuracy_score(y[te], bpred),
             baseline_tfidf_macro_f1=f1_score(y[te], bpred, average="macro"))
    print({k: round(v, 3) for k, v in m.items()})
    metrics[name] = {k: round(float(v), 4) for k, v in m.items()}
    return C, clf


def pick_threshold(X, Y):
    est = OneVsRestClassifier(LogisticRegression(C=8, max_iter=3000, class_weight="balanced"), n_jobs=-1)
    P = cross_val_predict(est, X, Y, cv=KFold(3, shuffle=True, random_state=SEED), method="predict_proba")
    best_t, best_f = 0.5, -1
    for t in np.arange(0.2, 0.96, 0.05):
        f = f1_score(Y, decide(P, t), average="micro")
        if f > best_f:
            best_t, best_f = float(t), f
    return round(best_t, 2)


def decide(P, t):
    D = (P >= t).astype(int)
    empty = D.sum(1) == 0
    D[empty, P[empty].argmax(1)] = 1          # always output at least one label
    return D


def multi_label(name, lists, labels, mask):
    mlb = MultiLabelBinarizer(classes=labels)
    Y = mlb.fit_transform(lists)
    tr, te = sel(train, mask), sel(test, mask)
    print(f"\n[{name}] train {len(tr)} / test {len(te)} | labels {len(labels)}")
    t = pick_threshold(R[tr], Y[tr])
    clf = OneVsRestClassifier(LogisticRegression(C=8, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(R[tr], Y[tr])
    P = clf.predict_proba(R[te])
    D = decide(P, t)
    top1 = float(np.mean([Y[te][i, P[i].argmax()] == 1 for i in range(len(te))]))
    prior = Y[tr].mean(0)
    base_D = decide(np.tile(prior, (len(te), 1)), 0.5)
    m = dict(threshold=t, micro_f1=f1_score(Y[te], D, average="micro"), macro_f1=f1_score(Y[te], D, average="macro", zero_division=0),
             top1_hit_rate=top1, baseline_prior_micro_f1=f1_score(Y[te], base_D, average="micro"))
    print({k: round(v, 3) for k, v in m.items()})
    metrics[name] = {k: round(float(v), 4) for k, v in m.items()}
    return t, mlb


# ----------------------------------------------------------------- evaluate on held-out reports
C_theme, _ = single_label("theme", df.cluster.values, has_theme)
C_prob, _ = single_label("primary_problem", y_prob, has_prob)
t_phase, mlb_phase = multi_label("flight_phase", phase_lists, phase_labels, has_phase)
t_anom, mlb_anom = multi_label("anomaly", anom_lists, anom_labels, has_anom)

# ----------------------------------------------------------------- novelty detection
tr_th = sel(train, has_theme)
ids_all = np.array(sorted(set(df.cluster[has_theme])))


def centroids(ix, ids):
    C = np.vstack([R[ix][df.cluster.values[ix] == c].mean(0) for c in ids])
    return C / np.linalg.norm(C, axis=1, keepdims=True)


C_tr = centroids(tr_th, ids_all)
maxsim = (R[test] @ C_tr.T).max(1)
in_theme = has_theme[test]
thr = float(np.percentile(maxsim[in_theme], 10))      # 90% of in-theme reports stay above it
metrics["novelty"] = dict(
    similarity_threshold=round(thr, 4),
    auc_inTheme_vs_unclustered=round(float(roc_auc_score(in_theme, maxsim)), 4),
    pct_inTheme_flagged=round(float((maxsim[in_theme] < thr).mean() * 100), 1),
    pct_unclustered_flagged=round(float((maxsim[~in_theme] < thr).mean() * 100), 1))
print("\n[novelty]", metrics["novelty"])

# approximate leave-one-theme-out: if a theme did not exist, would its reports look novel?
rows = []
for k, c in enumerate(ids_all):
    members = np.where(df.cluster.values == c)[0]
    others = np.delete(C_tr, k, axis=0)
    ms = (R[members] @ others.T).max(1)
    rows.append(dict(cluster=int(c), theme_name=names.get(int(c), ""), n_reports=len(members),
                     pct_flagged_novel=round(float((ms < thr).mean() * 100), 1)))
lto = pd.DataFrame(rows).sort_values("pct_flagged_novel", ascending=False)
lto.to_csv(f"{OUT_DIR}/leave_theme_out.csv", index=False)
metrics["novelty"]["leave_theme_out_mean_pct_flagged"] = round(float(lto.pct_flagged_novel.mean()), 1)
print("   leave-theme-out: mean % of a removed theme's reports flagged novel =", metrics["novelty"]["leave_theme_out_mean_pct_flagged"])

# ----------------------------------------------------------------- final fit on ALL data
print("\nFitting final models on all reports ...")
all_th, all_pr = np.where(has_theme)[0], np.where(has_prob)[0]
theme_clf = LogisticRegression(C=C_theme, max_iter=3000).fit(R[all_th], df.cluster.values[all_th])
prob_clf = LogisticRegression(C=C_prob, max_iter=3000).fit(R[all_pr], y_prob[all_pr])
all_ph, all_an = np.where(has_phase)[0], np.where(has_anom)[0]
phase_clf = OneVsRestClassifier(LogisticRegression(C=8, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(R[all_ph], mlb_phase.transform([phase_lists[i] for i in all_ph]))
anom_clf = OneVsRestClassifier(LogisticRegression(C=8, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(R[all_an], mlb_anom.transform([anom_lists[i] for i in all_an]))
C_all = centroids(all_th, ids_all)

art = dict(
    version=VERSION, embed_model=EMBED_MODEL, dim=int(R.shape[1]),
    theme=dict(clf=theme_clf, names={int(c): names.get(int(c), f"cluster {c}") for c in ids_all},
               centroids=C_all, centroid_ids=ids_all, novelty_sim_threshold=thr, low_conf_threshold=LOW_CONF),
    problem=dict(clf=prob_clf),
    phase=dict(clf=phase_clf, labels=phase_labels, threshold=t_phase),
    anomaly=dict(clf=anom_clf, labels=anom_labels, threshold=t_anom),
    reference=dict(vectors=R.astype(np.float16), acn=df.get("acn", df.report_id).tolist(),
                   synopsis=[s[:160] for s in df.synopsis], cluster=df.cluster.tolist()),
    metrics=metrics)
path = f"{OUT_DIR}/asrs_theme_model_v1.joblib"
joblib.dump(art, path, compress=3)
json.dump(metrics, open(f"{OUT_DIR}/metrics.json", "w"), indent=2)

m = metrics
card = f"""# Model card: {VERSION}

**What it does.** Reads the narrative of an aviation safety report and returns: a theme (one of {len(ids_all)} data-driven
themes found in NASA ASRS reports), a novelty flag, an ASRS-style primary problem, flight phase(s) and anomaly categories,
with confidence scores and the 3 most similar ASRS reports.

**Trained on.** {n} NASA ASRS reports, June 2025 - June 2026 (voluntary, mostly minor incidents, US-centred, first-person narratives).
Embeddings: {EMBED_MODEL}. Classifiers: logistic regression on the embeddings.

**Held-out results (20% of ASRS reports, never used for training).**
- Theme: accuracy {m['theme']['accuracy']}, macro-F1 {m['theme']['macro_f1']}, top-3 accuracy {m['theme']['top3_accuracy']} (TF-IDF baseline accuracy {m['theme']['baseline_tfidf_accuracy']}).
- Primary problem: accuracy {m['primary_problem']['accuracy']}, macro-F1 {m['primary_problem']['macro_f1']} (TF-IDF baseline {m['primary_problem']['baseline_tfidf_accuracy']}, majority {m['primary_problem']['baseline_majority_accuracy']}).
- Flight phase: micro-F1 {m['flight_phase']['micro_f1']}, top-1 hit rate {m['flight_phase']['top1_hit_rate']}.
- Anomaly categories: micro-F1 {m['anomaly']['micro_f1']}, top-1 hit rate {m['anomaly']['top1_hit_rate']}.
- Novelty flag: separates in-theme from unclustered ASRS reports with AUC {m['novelty']['auc_inTheme_vs_unclustered']}; flags {m['novelty']['pct_unclustered_flagged']}% of unclustered and {m['novelty']['pct_inTheme_flagged']}% of in-theme reports.
  Approximate leave-one-theme-out: on average {m['novelty']['leave_theme_out_mean_pct_flagged']}% of a removed theme's reports are flagged novel.

**Limits (read before using).**
- Themes, primary problem, phase and anomaly are ASRS-STYLE labels predicted from text. They are not official AAIB/CAAM classifications.
- Not validated on Malaysian reports yet. AAIB reports are long, formal and cover serious accidents, so expect a shift. Always check `needs_review` rows.
- Labels are only as good as the ASRS reporters' own classification; themes are unsupervised and some are broad or overlapping.
- English text only. Not for operational safety decisions without expert review.
- The novelty threshold was set on the held-out set (so it is an operating point, not an independent test).
"""
open(f"{OUT_DIR}/model_card.md", "w").write(card)
print(f"\nSaved {path} ({os.path.getsize(path) / 1e6:.1f} MB), metrics.json, model_card.md, leave_theme_out.csv")
