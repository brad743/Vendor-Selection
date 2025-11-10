#!/usr/bin/env python3
"""Improved Vendor-Requirement Matching

Usage in Google Colab:
- Upload two CSVs (vendors.csv, requirements.csv)
- Run this script cell (or import as module)
- Adjust parameters (TFIDF_THRESHOLD, FUZZY_THRESHOLD, SHORTLIST_PCT)

Dependencies:
pip install -q rapidfuzz scikit-learn
"""

import io
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz

# Try to import Colab files helper; safe fallback when not in Colab
try:
    from google.colab import files
    _HAS_COLAB = True
except Exception:
    _HAS_COLAB = False

# --- Configurable thresholds ---
TFIDF_THRESHOLD = 0.28        # cosine similarity threshold (0-1)
FUZZY_THRESHOLD = 75          # rapidfuzz token_set_ratio threshold (0-100)
SHORTLIST_PCT = 45.0          # percent threshold for shortlist

# --- Utilities ---
def upload_two_files(prompt="Upload vendor CSV then requirements CSV"):
    if not _HAS_COLAB:
        raise RuntimeError("google.colab.files not available in this environment. Run in Colab or adapt file input.")
    print("📂", prompt)
    uploaded = files.upload()
    names = list(uploaded.keys())
    if len(names) < 2:
        raise ValueError("Please upload two files. Uploaded: " + ", ".join(names))
    return uploaded, names


def read_csv_from_uploaded(uploaded, name):
    return pd.read_csv(io.BytesIO(uploaded[name]))

def guess_column(df, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

def parse_bool_like(val):
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in ("yes","y","true","t","1","required","must","mandatory","req")

def normalize_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^\w\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Main routine ---
def main():
    uploaded, names = upload_two_files()
    df1 = read_csv_from_uploaded(uploaded, names[0])
    df2 = read_csv_from_uploaded(uploaded, names[1])

    vendor_candidates = ["vendor", "vendor name", "company", "supplier", "vendor_name", "organization"]
    req_candidates = ["requirement", "essential criteria", "criteria", "essential", "requirement title", "requirement_desc", "requirement description"]

    vendor_name_col = guess_column(df1, vendor_candidates) or guess_column(df2, vendor_candidates)
    req_title_col = guess_column(df1, req_candidates) or guess_column(df2, req_candidates)
    req_required_col = guess_column(df1, ["required", "mandatory", "must"]) or guess_column(df2, ["required", "mandatory", "must"])

    if vendor_name_col and vendor_name_col in df1.columns:
        vendor_df = df1.copy()
        req_df = df2.copy()
    elif vendor_name_col and vendor_name_col in df2.columns:
        vendor_df = df2.copy()
        req_df = df1.copy()
    else:
        print("⚠️ Couldn't confidently detect which file is vendors vs requirements. Assuming first file is vendors.")
        vendor_df = df1.copy()
        req_df = df2.copy()

    print("Detected columns (best-effort):")
    print(" vendor name:", vendor_name_col)
    print(" requirement title:", req_title_col)
    print(" required flag column:", req_required_col)

    if req_title_col is None:
        text_cols = [c for c in req_df.columns if req_df[c].dtype == object or req_df[c].dtype == "string"]
        if text_cols:
            req_title_col = text_cols[0]
            print(f"Using '{req_title_col}' from requirements file as requirement title.")
        else:
            raise ValueError("No text column found in the requirements file to use as requirement title.")

    if req_required_col and req_required_col in req_df.columns:
        required_mask = req_df[req_required_col].apply(parse_bool_like)
        req_df = req_df[required_mask]

    req_df[req_title_col] = req_df[req_title_col].fillna("\"").astype(str)
    requirements = [normalize_text(s) for s in req_df[req_title_col].tolist() if s.strip() != ""]
    total_reqs = len(requirements)
    print(f"Total requirements to evaluate: {total_reqs}")

    vendor_df = vendor_df.fillna("").astype(str)
    vendor_name_col = vendor_name_col if vendor_name_col in vendor_df.columns else vendor_df.columns[0]
    def vendor_blob(row):
        return normalize_text(" ".join(row.values))
    vendor_texts = vendor_df.apply(vendor_blob, axis=1).tolist()

    # --- TF-IDF vectorization and cosine similarity ---
    docs = requirements + vendor_texts
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vectorizer.fit_transform(docs)
    req_vecs = normalize(X[:len(requirements)], axis=1)
    vendor_vecs = normalize(X[len(requirements):], axis=1)

    cosine_sim = req_vecs.dot(vendor_vecs.T).toarray()

    matched_counts = []
    matched_examples = []
    missing_examples = []
    match_matrix = []

    for j, vtext in enumerate(vendor_texts):
        matched_for_vendor = []
        for i, req in enumerate(requirements):
            tfidf_score = cosine_sim[i, j]
            fuzzy_score = fuzz.token_set_ratio(req, vtext)
            is_matched = (tfidf_score >= TFIDF_THRESHOLD) or (fuzzy_score >= FUZZY_THRESHOLD)
            if is_matched:
                matched_for_vendor.append((i, req, tfidf_score, fuzzy_score))
        matched_counts.append(len(matched_for_vendor))
        matched_for_vendor.sort(key=lambda tup: (tup[2], tup[3]), reverse=True)
        matched_examples.append([t[1] for t in matched_for_vendor[:5]])
        missing = [requirements[i] for i in range(len(requirements)) if all(i != m[0] for m in matched_for_vendor)]
        missing_examples.append(missing[:5])
        match_matrix.append([1 if any(m[0] == i for m in matched_for_vendor) else 0 for i in range(len(requirements))])

    vendor_df["Matched_Count"] = matched_counts
    vendor_df["Total_Req"] = total_reqs
    vendor_df["Match %"] = vendor_df["Matched_Count"] / vendor_df["Total_Req"] * 100
    vendor_df["Matched_Examples"] = [", ".join(x) if x else "" for x in matched_examples]
    vendor_df["Missing_Examples"] = [", ".join(x) if x else "" for x in missing_examples]

    shortlist = vendor_df[vendor_df["Match %"] >= SHORTLIST_PCT].copy()
    shortlist = shortlist.sort_values(by="Match %", ascending=False)

    print(f"\nShortlist (Match % >= {SHORTLIST_PCT}): {len(shortlist)} vendors")
    display_cols = [vendor_name_col, "Match %", "Matched_Count", "Matched_Examples", "Missing_Examples"]
    try:
        from IPython.display import display
        display(shortlist[display_cols].head(50))
    except Exception:
        print(shortlist[display_cols].head(50).to_string(index=False))

    out_file = "shortlist_results.csv"
    shortlist.to_csv(out_file, index=False)
    vendor_df.to_csv("all_vendors_with_matches.csv", index=False)
    print(f"\nSaved: {out_file} and all_vendors_with_matches.csv")
    if _HAS_COLAB:
        try:
            files.download(out_file)
        except Exception:
            print("Files ready in the session storage.")


if __name__ == "__main__":
    main()
