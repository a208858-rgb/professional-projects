"""
ASRS pipeline: serious (1) vs minor (0) event triage + topics + weak signals.

Install:
    pip install pandas scikit-learn matplotlib bertopic sentence-transformers transformers torch

Run (input = the merged CSV from merge_asrs.py):
    python asrs_pipeline.py                       # eda + baseline + topics
    python asrs_pipeline.py --stages bert         # DistilBERT (use a GPU / Colab)
    python asrs_pipeline.py --stages eda baseline topics bert

Outputs go to ./outputs/
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

CSV = "asrs_2025-06_to_2026-06.csv"
OUT = "outputs"
SEED = 42

# 1) LABEL: a report is "serious" if its RESULT/outcome fields mention any of these.
#    Check the real values in your CSV and edit this list.
SERIOUS_LABEL = (r"emergency|evasive|rejected takeoff|go around|missed approach|"
                 r"injur|aircraft damage|evacuat")

# 2) LEAKAGE MASK: phrases removed from the narrative so the model can't just
#    read the answer. Mirrors the label, plus common wording variants.
MASK = (r"declared? (an )?emergency|emergency|evasive|go[- ]?around|missed approach|"
        r"reject(ed)? (the )?take ?off|\brto\b|injur\w*|evacuat\w*")

ABBREV = {"atc": "air traffic control", "twr": "tower", "app": "approach",
          "acft": "aircraft", "rwy": "runway", "twy": "taxiway", "capt": "captain",
          "f/o": "first officer", "fo": "first officer", "tcas": "traffic collision avoidance",
          "ra": "resolution advisory", "ta": "traffic advisory", "alt": "altitude"}


# ---------------------------------------------------------------- helpers
def find_col(df, *keys):
    for c in df.columns:
        if all(k in c for k in keys):
            return c
    return None


def clean_text(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    words = [ABBREV.get(w, w) for w in s.split(" ")]
    return " ".join(words).strip()


def load():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV, dtype=str).fillna("")

    narr, syn = find_col(df, "narrative"), find_col(df, "synopsis")
    if not narr:
        raise SystemExit(f"No narrative column found. Columns: {list(df.columns)}")
    raw = df[narr] + " " + (df[syn] if syn else "")
    df["text_clean"] = raw.map(clean_text)
    df["text"] = df["text_clean"].str.replace(MASK, " [mask] ", regex=True)

    res_cols = [c for c in df.columns if "result" in c]
    if not res_cols:
        raise SystemExit(f"No 'result' columns found for the label. Columns: {list(df.columns)}")
    blob = df[res_cols].agg(" ; ".join, axis=1).str.lower()
    df["y"] = blob.str.contains(SERIOUS_LABEL, regex=True).astype(int)

    if "event_month" in df:
        df["event_month"] = pd.to_datetime(df["event_month"], errors="coerce")
    else:
        d = find_col(df, "date")
        df["event_month"] = pd.to_datetime(df[d].str[:6], format="%Y%m", errors="coerce")

    df = df[df["text"].str.len() > 30].reset_index(drop=True)
    print(f"{len(df)} reports | serious = {df['y'].mean():.1%} | result columns used: {res_cols}")
    return df


def split(df):
    return train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)


def report(name, y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    m = dict(model=name, roc_auc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob),
             f1=f1_score(y, pred), recall=recall_score(y, pred),
             precision=precision_score(y, pred, zero_division=0))
    print(f"\n=== {name} ===")
    print(classification_report(y, pred, digits=3))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    json.dump(m, open(f"{OUT}/metrics_{name}.json", "w"), indent=2)
    ConfusionMatrixDisplay.from_predictions(y, pred)
    plt.title(name)
    plt.savefig(f"{OUT}/cm_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return m


def barh(series, title, fname, n=12):
    series.head(n)[::-1].plot.barh(figsize=(7, 4))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=150)
    plt.close()


# ---------------------------------------------------------------- stage 1: EDA
def run_eda(df):
    barh(df["event_month"].dt.strftime("%Y-%m").value_counts().sort_index()[::-1],
         "Reports per month (latest months are incomplete)", "eda_months.png", n=20)
    for key, fname in [("anomaly", "eda_anomaly.png"), ("phase", "eda_phase.png")]:
        c = find_col(df, key)
        if c:
            vc = df[c].str.split(";").explode().str.strip().replace("", np.nan).dropna().value_counts()
            barh(vc, f"Top values: {c}", fname)
    print("Label balance:\n", df["y"].value_counts())
    print("Text length (words):\n", df["text"].str.split().str.len().describe())


# ---------------------------------------------------------------- stage 2: baseline
def tfidf_lr(train, test, col):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          stop_words="english", max_features=50000)
    Xtr, Xte = vec.fit_transform(train[col]), vec.transform(test[col])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(Xtr, train["y"])
    return vec, clf, clf.predict_proba(Xte)[:, 1]


def run_baseline(df):
    tr, te = split(df)
    # leakage check: same model on unmasked text. A big jump = the label is readable from keywords.
    _, _, p_leak = tfidf_lr(tr, te, "text_clean")
    print("Leak check (unmasked) ROC-AUC:", round(roc_auc_score(te["y"], p_leak), 3))

    vec, clf, prob = tfidf_lr(tr, te, "text")
    report("tfidf_logreg_masked", te["y"], prob)

    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    pd.DataFrame({"top_serious": names[order[-25:]][::-1],
                  "top_minor": names[order[:25]]}).to_csv(f"{OUT}/top_words.csv", index=False)

    # error analysis: serious reports the model missed -> read these manually
    te = te.assign(prob=prob)
    te[(te.y == 1) & (te.prob < 0.5)].sort_values("prob").head(50)[["prob", "text_clean"]] \
        .to_csv(f"{OUT}/missed_serious_reports.csv", index=False)


# ---------------------------------------------------------------- stage 3: topics + weak signals
def run_topics(df):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    docs = df["text_clean"].tolist()
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, show_progress_bar=True)
    model = BERTopic(min_topic_size=15)
    df["topic"], _ = model.fit_transform(docs, emb)
    info = model.get_topic_info()
    info.to_csv(f"{OUT}/topics.csv", index=False)
    df[["event_month", "topic", "y"]].to_csv(f"{OUT}/report_topics.csv", index=False)
    print(info[["Topic", "Count", "Name"]].head(15))

    # which topics are most "serious"?
    sev = df[df.topic >= 0].groupby("topic")["y"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    sev.merge(info[["Topic", "Name"]], left_index=True, right_on="Topic").to_csv(f"{OUT}/topic_severity.csv", index=False)

    # weak signals (exploratory with 1 year): topic SHARE per month, so incomplete months don't mislead
    ct = pd.crosstab(df["event_month"], df["topic"]).drop(columns=-1, errors="ignore")
    share = ct.div(ct.sum(axis=1), axis=0).sort_index()
    base, recent = share.iloc[:-3], share.iloc[-2:].mean()
    z = (recent - base.mean()) / base.std().clip(lower=0.005)
    flag = pd.DataFrame({"z": z, "recent_share": recent, "base_share": base.mean()})
    flag = flag[(flag.z > 2) & (flag.recent_share > flag.base_share)].sort_values("z", ascending=False)
    flag.merge(info[["Topic", "Name"]], left_index=True, right_on="Topic") \
        .to_csv(f"{OUT}/weak_signals.csv", index=False)
    print("\nFlagged emerging topics:\n", flag.head(10))


# ---------------------------------------------------------------- stage 4: DistilBERT
def run_bert(df):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tr, te = split(df)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class DS(torch.utils.data.Dataset):
        def __init__(self, d):
            self.enc = tok(d["text"].tolist(), truncation=True, max_length=256, padding=True)
            self.y = d["y"].tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    pos = tr["y"].mean()
    w = torch.tensor([1 / (1 - pos), 1 / pos], dtype=torch.float)
    w = w / w.sum()

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(output_dir=f"{OUT}/bert_ckpt", num_train_epochs=3,
                             per_device_train_batch_size=16, learning_rate=2e-5,
                             save_strategy="no", report_to="none", seed=SEED)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=DS(tr))
    trainer.train()
    logits = trainer.predict(DS(te)).predictions
    prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    report("distilbert_masked", te["y"], prob)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["eda", "baseline", "topics"],
                    choices=["eda", "baseline", "topics", "bert"])
    stages = ap.parse_args().stages
    data = load()
    if "eda" in stages:
        run_eda(data)
    if "baseline" in stages:
        run_baseline(data)
    if "topics" in stages:
        run_topics(data)
    if "bert" in stages:
        run_bert(data)
    print(f"\nDone. See ./{OUT}/")
