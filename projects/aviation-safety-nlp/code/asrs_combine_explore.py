# %% [markdown]
# # ASRS: combine 13 monthly CSVs into ONE file, then explore it
# Works as a normal script (python asrs_combine_explore.py) or in Jupyter / Colab
# (each "# %%" block is a cell).
#
# Setup: put your 13 CSVs in a folder called asrs_raw/ next to this file.
#   pip install pandas matplotlib

# %% Settings
import glob
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

INPUT_DIR = "asrs_raw"
COMBINED_CSV = "asrs_combined.csv"
OUT = "explore_outputs"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.max_columns", 50, "display.width", 200, "display.max_colwidth", 80)


# %% 1. Read every CSV (handles ASRS's extra header row) and combine
def read_asrs_csv(path):
    peek = pd.read_csv(path, header=None, nrows=3, dtype=str,
                       encoding="utf-8", encoding_errors="replace")
    header_row = 0
    for i in range(len(peek)):
        if "ACN" in " ".join(peek.iloc[i].fillna("").astype(str)).upper():
            header_row = i
            break
    df = pd.read_csv(path, header=header_row, dtype=str,
                     encoding="utf-8", encoding_errors="replace")
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^0-9A-Za-z_]", "", regex=True).str.lower())
    df["source_file"] = os.path.basename(path)
    return df


files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
print(f"Found {len(files)} files (expected 13)")
frames = []
for f in files:
    d = read_asrs_csv(f)
    print(f"{os.path.basename(f):28s} {len(d):6d} rows, {d.shape[1]} cols")
    frames.append(d)

df = pd.concat(frames, ignore_index=True, sort=False).dropna(how="all")

# %% 2. Remove duplicate reports, save ONE csv
acn = next((c for c in df.columns if c.startswith("acn")), None)
if acn:
    n0 = len(df)
    df = df.drop_duplicates(subset=acn).reset_index(drop=True)
    print(f"\nDuplicates removed: {n0 - len(df)}")

df.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} reports -> {COMBINED_CSV}")

# %% 3. First look
print("\nShape:", df.shape)
print("\nColumns:")
for i, c in enumerate(df.columns):
    print(f"{i:3d}  {c}")
print("\nSample rows:")
print(df.head(3).T.head(40))

# %% 4. Missing values
miss = (df.replace("", pd.NA).isna().mean() * 100).sort_values(ascending=False)
print("\n% missing per column (top 25):")
print(miss.head(25).round(1))
miss.head(25)[::-1].plot.barh(figsize=(7, 6), title="% missing")
plt.tight_layout()
plt.savefig(f"{OUT}/missing.png", dpi=150)
plt.close()

# %% 5. Reports per month
date_col = next((c for c in df.columns if "date" in c), None)
if date_col:
    df["event_month"] = pd.to_datetime(df[date_col].astype(str).str[:6],
                                       format="%Y%m", errors="coerce")
    monthly = df["event_month"].dt.strftime("%Y-%m").value_counts().sort_index()
    print("\nReports per month:\n", monthly)
    monthly.plot.bar(figsize=(9, 4), title="Reports per month (latest months incomplete)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/reports_per_month.png", dpi=150)
    plt.close()


# %% 6. Top values for the key categorical columns
# ASRS cells often hold several values separated by ";", so split before counting
def top_values(col, n=15):
    s = (df[col].fillna("").str.split(";").explode().str.strip()
         .replace("", pd.NA).dropna())
    return s.value_counts().head(n)


keywords = ["anomaly", "phase", "aircraft", "result", "contributing",
            "primary_problem", "locale", "state", "reporter", "make_model", "operator"]
for kw in keywords:
    for col in [c for c in df.columns if kw in c][:2]:
        vc = top_values(col)
        if len(vc):
            print(f"\n--- {col} ---\n{vc}")
            vc[::-1].plot.barh(figsize=(8, 4), title=col)
            plt.tight_layout()
            plt.savefig(f"{OUT}/top_{col[:40]}.png", dpi=150)
            plt.close()

# %% 7. Narrative text exploration
narr = next((c for c in df.columns if "narrative" in c), None)
if narr:
    words = df[narr].fillna("").str.split().str.len()
    print("\nNarrative length (words):\n", words.describe().round(1))
    words.plot.hist(bins=40, figsize=(7, 4), title="Narrative length (words)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/narrative_length.png", dpi=150)
    plt.close()

    stop = set("the a an and or of to in on at for with was were is it that this i we he she they "
               "by from as be had have not but our my then after when so out up no all as are "
               "been just would could them their there what which while into over about back".split())
    tokens = re.findall(r"[a-z]{3,}", " ".join(df[narr].fillna("").str.lower()))
    common = Counter(t for t in tokens if t not in stop).most_common(30)
    print("\nMost common words in narratives:", common)

    print("\nThree random narratives:")
    for t in df[narr].dropna().sample(3, random_state=1):
        print("-", t[:500], "...\n")

# %% 8. Cross-tab example: anomaly vs flight phase (edit the keywords if needed)
a = next((c for c in df.columns if "anomaly" in c), None)
p = next((c for c in df.columns if "phase" in c), None)
if a and p:
    x = df[[a, p]].copy()
    x[a] = x[a].fillna("").str.split(";").str[0].str.strip()   # first anomaly only
    x[p] = x[p].fillna("").str.split(";").str[0].str.strip()
    top_a = x[a].value_counts().head(8).index
    top_p = x[p].value_counts().head(6).index
    ct = pd.crosstab(x[x[a].isin(top_a)][a], x[x[p].isin(top_p)][p])
    print("\nAnomaly vs flight phase (first value of each):\n", ct)
    ct.to_csv(f"{OUT}/anomaly_vs_phase.csv")

print(f"\nDone. Combined file: {COMBINED_CSV} | charts and tables: ./{OUT}/")
