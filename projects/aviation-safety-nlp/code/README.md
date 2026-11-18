# Aviation Safety Incident Mining (NASA ASRS) and Malaysian Report Classifier

**Research question.** Can semantic clustering of NASA ASRS narratives identify coherent, interpretable themes in
aviation safety events *without* predefined incident categories, and can those themes be reused to organise
Malaysian aviation safety reports (AAIB PDFs) into a CSV over time?

**Data.** 13 monthly ASRS exports (June 2025 to June 2026) -> 6,314 cleaned reports. ASRS is voluntary reporting, so
results describe *reported* patterns, not true incident rates.

---------------------------------------------------------------------------------------------------------------

## Pipeline at a glance

```
ASRS CSV exports (13)
   |  merge_asrs.py / asrs_combine_explore.py          -> asrs_combined.csv (+ first exploration)
   v
step1_prepare_text.py                                  -> asrs_clean.csv, asrs_chunks.csv
   v
step2_compare_embeddings.py                            -> results_embedding_comparison.csv   (picked MPNet)
   v
step3_cluster_tuning.py                                -> emb_full_mpnet.npy, results_tuning.csv, topics.csv, ...
   v
step3b_finalize.py                                     -> asrs_with_clusters_v2.csv, topics_v2.csv, similar_clusters.csv
   v   (edit cluster_names.csv by hand)
train_theme_model.py                                   -> models/asrs_theme_model_v1.joblib, metrics.json, model_card.md
   v
classify_pdfs.py  (Malaysian PDFs in ./reports)        -> malaysia_reports_classified.csv  (grows each run)
```
`asrs_text.py` is shared cleaning/chunking code used by training and prediction so both treat text identically.

## Install
```
pip install -r requirements.txt
```
(`hdbscan` can fail to build on Windows; use Google Colab if so. `gensim` is optional.)

---------------------------------------------------------------------------------------------------------------

## Scripts, in run order

| # | Script | Input | Output | What it does |
|---|--------|-------|--------|--------------|
| 0a | `merge_asrs.py` | `asrs_raw/*.csv` | `asrs_2025-06_to_2026-06.csv` | Merges monthly exports (handles ASRS's extra header row), removes duplicate ACNs, keeps the date window. |
| 0b | `asrs_combine_explore.py` | `asrs_raw/*.csv` | `asrs_combined.csv`, `explore_outputs/` | Same merge, then first exploration: missing values, reports per month, top anomalies / phases / aircraft, narrative length, common words. **Use this one.** |
| 1 | `step1_prepare_text.py` | `asrs_combined.csv` | `asrs_clean.csv`, `asrs_chunks.csv` | Expands ASRS shorthand (ATC, RWY, F/O...), replaces hidden-name tokens (ZZZ), lowercases, drops reports under 20 words and duplicate narratives, splits long narratives into ~200-word chunks (embedding models truncate at ~256 tokens). |
| 2 | `step2_compare_embeddings.py` | `asrs_clean.csv`, `asrs_chunks.csv` | `results_embedding_comparison.csv`, `emb_*.npy` | Benchmarks MiniLM, BGE-small, MPNet on a 2,000-report sample. Metric: how often a report's 10 nearest neighbours share its expert label (primary problem, flight phase, anomaly-set overlap) versus chance. Labels are used only for scoring. |
| 3 | `step3_cluster_tuning.py` | `asrs_clean.csv`, `asrs_chunks.csv` | `emb_full_mpnet.npy`, `results_tuning.csv`, `results_top_configs.csv`, `asrs_with_clusters.csv`, `topics.csv`, `cluster_map.png` | Embeds all reports with MPNet; grid-searches 72 UMAP + HDBSCAN settings; keeps only interpretable ones (10-80 clusters, <=45% unclustered, no cluster >25%); scores the top 3 on validity, coherence and stability (re-clustering 80% subsamples). |
| 3b | `step3b_finalize.py` | `asrs_clean.csv`, `emb_full_mpnet.npy` | `asrs_with_clusters_v2.csv`, `topics_v2.csv`, `similar_clusters.csv`, `cluster_map_v2.png` | Re-runs the most stable setting (step 3's tie-break had picked the least stable), cleans cluster keywords, computes NPMI coherence without gensim, soft-assigns unclustered reports (separate column), lists near-duplicate clusters. |
| - | `cluster_names.csv` | (edit by hand) | - | Human-readable names for the 38 themes. Drafts only; check each against `topics_v2.csv`. |
| 4a | `train_theme_model.py` | `asrs_with_clusters_v2.csv`, `emb_full_mpnet.npy`, `cluster_names.csv` | `models/asrs_theme_model_v1.joblib`, `metrics.json`, `model_card.md`, `leave_theme_out.csv` | Trains classifiers on report vectors: theme, primary problem, flight phase (multi-label), anomaly categories (multi-label). Evaluates on 20% held-out reports against TF-IDF and majority baselines. Builds a novelty detector (distance to theme centres) and an approximate leave-one-theme-out test. |
| 4b | `classify_pdfs.py` | `reports/*.pdf`, the model | `malaysia_reports_classified.csv` | Reads AAIB-style PDFs, removes headers/footers, extracts synopsis + history of the flight, pulls report facts (type, number, date, operator, aircraft, registration, location, fatalities, ADREP category, AAIB cause text), predicts theme / primary problem / phase / anomaly with confidence and novelty flag, lists the 3 nearest ASRS reports, flags rows for review, and appends to the CSV (skips files already done; `--force` redoes all but keeps your `review_*` columns). |
| - | `asrs_pipeline.py` | `asrs_combined`-style CSV | `outputs/` | **Early experiment, superseded.** Serious-vs-minor classifier with leakage masking, plus a first BERTopic run and a crude weak-signal detector. Kept for reference; set `CSV` at the top to your file name. |

Typical commands:
```
python asrs_combine_explore.py
python step1_prepare_text.py
python step2_compare_embeddings.py
python step3_cluster_tuning.py       # ~20-30 min
python step3b_finalize.py            # ~3-5 min
python train_theme_model.py          # ~5-10 min
python classify_pdfs.py              # after putting PDFs in ./reports
```

---------------------------------------------------------------------------------------------------------------

## Results so far (real runs on the ASRS data)

* **Data:** 6,314 reports kept (56 under 20 words and 2 duplicates dropped); median 190 words, 47% longer than 200 words (hence chunking).
* **Embeddings** (neighbour agreement; chance in brackets): MPNet best on primary problem 0.556 (0.241) and anomaly overlap 0.561 (0.224); flight phase 0.356 (0.111). MiniLM 0.537 / 0.538 / 0.350. BGE-small worst (0.499 / 0.494 / 0.311). A TF-IDF keyword baseline scored about 0.52 / 0.55 / 0.38, so the gain from embeddings is modest, and TF-IDF is better on flight phase.
* **Clustering:** 38 themes, 23% of reports unclustered, largest cluster 9% of reports, stability (ARI over 80% subsamples) 0.91 for the final setting (0.60 for the setting step 3 had picked by mistake), mean NPMI coherence 0.24 (range 0.02-0.48), mean 63% agreement with primary problem against 34% for the largest class.
* **Themes found without labels:** drones/UAS authorization, GPS jamming/spoofing, door and slide arming, pressurization, smoke and fumes, pushback/tug, taxiway confusion, wake turbulence, TCAS advisories, hydraulics, engine power loss, icing and more.

## Not done yet (do not claim these)

* `train_theme_model.py` and `classify_pdfs.py` were tested on stand-in vectors and a synthetic AAIB-style PDF, not on real MPNet outputs or real AAIB PDFs. Run training and read `models/metrics.json` before quoting any accuracy.
* No LDA/NMF baseline yet, so the coherence score has no comparison.
* No human interpretability rating (word-intrusion / theme rating with a second rater).
* No validation on Malaysian reports (needs ~30-50 hand-labelled reports: fill the `review_*` columns).
* Trends over time are not reliable: June 2026 is incomplete (publication lag) and one year is short.

## Known limits

* ASRS is voluntary and mostly minor events; AAIB reports are long, formal and cover serious accidents, so a distribution shift is expected and many Malaysian reports may be flagged "novel".
* Predicted primary problem / phase / anomaly are ASRS-style labels, not official AAIB or CAAM classifications.
* Scanned PDFs without a text layer are flagged for OCR but not read. English text only; possible Malay is flagged.
* Themes were found unsupervised; some are broad (e.g. approach/altitude) or near-duplicates (two pushback themes, two taxiway themes).

## Next steps

1. Run `train_theme_model.py`; read `metrics.json` and `leave_theme_out.csv`.
2. Add the LDA/NMF baseline and a human rating sheet (Step 4 validation).
3. Merge near-duplicate themes using `similar_clusters.csv`; finalize `cluster_names.csv`.
4. Download 10-20 public AAIB reports, run `classify_pdfs.py`, correct the `review_*` columns, measure transfer.
5. Optional: Streamlit demo, GitHub README with the results table.
