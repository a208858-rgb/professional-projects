"""
Merge monthly NASA ASRS CSV exports (asrs_YYYYMM.csv) into ONE CSV file.

Usage:
    1. Put all monthly files in one folder, e.g. ./asrs_raw/
    2. Run:  python merge_asrs.py
    3. Output: asrs_2025-06_to_2026-06.csv
"""

import glob
import os

import pandas as pd

INPUT_DIR = "asrs_raw"                       # folder with asrs_YYYYMM.csv files
OUTPUT_FILE = "asrs_2025-06_to_2026-06.csv"  # final single CSV


def read_asrs_csv(path):
    """Read one ASRS CSV. ASRS exports often have a group-label row above the
    real column names, so detect which row holds the headers."""
    peek = pd.read_csv(path, header=None, nrows=3, dtype=str,
                       encoding="utf-8", encoding_errors="replace")
    header_row = 0
    for i in range(len(peek)):
        row_text = " ".join(peek.iloc[i].fillna("").astype(str)).upper()
        if "ACN" in row_text:
            header_row = i
            break

    df = pd.read_csv(path, header=header_row, dtype=str,
                     encoding="utf-8", encoding_errors="replace")

    # tidy column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]", "", regex=True)
        .str.lower()
    )
    df["source_file"] = os.path.basename(path)
    return df


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in '{INPUT_DIR}/'")

    frames = []
    for f in files:
        try:
            df = read_asrs_csv(f)
            print(f"{os.path.basename(f):25s} {len(df):6d} rows")
            frames.append(df)
        except Exception as e:
            print(f"FAILED  {os.path.basename(f)}: {e}")

    data = pd.concat(frames, ignore_index=True, sort=False)

    # drop fully empty rows, then duplicate reports (same ACN)
    data = data.dropna(how="all")
    acn_col = next((c for c in data.columns if c.startswith("acn")), None)
    if acn_col:
        before = len(data)
        data = data.drop_duplicates(subset=acn_col)
        print(f"\nRemoved {before - len(data)} duplicate reports (by {acn_col})")

    # parse the event date (ASRS uses YYYYMM) and keep the target window
    date_col = next((c for c in data.columns if "date" in c), None)
    if date_col:
        data["event_month"] = pd.to_datetime(
            data[date_col].astype(str).str[:6], format="%Y%m", errors="coerce"
        )
        data = data[
            (data["event_month"] >= "2025-06-01")
            & (data["event_month"] <= "2026-06-30")
        ]
        print("\nReports per month:")
        print(data["event_month"].dt.strftime("%Y-%m").value_counts().sort_index())

    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(data)} reports, {data.shape[1]} columns -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
