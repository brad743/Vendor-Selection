#!/usr/bin/env python3
"""
Streamlit Shortlist Tool

Save this file as apps/streamlit_shortlist.py and run:
    pip install streamlit pandas scikit-learn rapidfuzz
    streamlit run apps/streamlit_shortlist.py
"""
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz

st.set_page_config(page_title="Booking System Shortlist Tool", layout="wide")
st.title("Booking System Shortlist Tool 🧩")
st.write("Upload your vendor list and requirements checklist to generate a shortlist.")

# Sidebar hyperparameters
st.sidebar.header("Matching settings")
TFIDF_THRESHOLD = st.sidebar.slider("TF‑IDF cosine threshold", 0.0, 1.0, 0.28, 0.01)
FUZZY_THRESHOLD = st.sidebar.slider("Fuzzy (token_set_ratio) threshold", 0, 100, 75)
SHORTLIST_PCT = st.sidebar.slider("Shortlist threshold (%)", 0, 100, 65)
use_required_flag = st.sidebar.checkbox("Filter to 'required' requirements only (if present)", value=True)
show_examples = st.sidebar.checkbox("Show matched / missing examples in table", value=True)

vendor_file = st.file_uploader("📁 Upload Vendor List CSV", type=["csv"])
req_file = st.file_uploader("Upload Requirements Checklist CSV", type=["csv"])/
# Utilities
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

def build_vendor_blob(df_row):
    return normalize_text(" ".join(df_row.values.astype(str)))

def run_matching(vendor_df: pd.DataFrame, req_df: pd.DataFrame,
                 req_title_col: str, req_required_col: str | None,
                 tfidf_th: float, fuzzy_th: int):
    # Filter required requirements if requested and flag exists
    if req_required_col and req_required_col in req_df.columns and use_required_flag:
        required_mask = req_df[req_required_col].apply(parse_bool_like)
        req_df = req_df[required_mask].copy()

    # Prepare requirements
    req_df[req_title_col] = req_df[req_title_col].fillna("").astype(str)
    requirements = [normalize_text(s) for s in req_df[req_title_col].tolist() if s.strip() != ""]
    total_reqs = len(requirements)

    if total_reqs == 0:
        return vendor_df.assign(Matched_Count=0, Total_Req=0, **{"Match %": 0.0})

    # Prepare vendor text blobs
    vendor_df = vendor_df.fillna("-").astype(str)
    vendor_texts = vendor_df.apply(build_vendor_blob, axis=1).tolist()

    # TF-IDF vectorization
    docs = requirements + vendor_texts
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vectorizer.fit_transform(docs)
    req_vecs = normalize(X[:len(requirements)], axis=1)
    vendor_vecs = normalize(X[len(requirements):], axis=1)
    cosine_sim = req_vecs.dot(vendor_vecs.T).toarray()  # shape (num_reqs, num_vendors)

    matched_counts = []
    matched_examples = []
    missing_examples = []
    matched_matrix = []

    for j, vtext in enumerate(vendor_texts):
        matched_for_vendor = []
        for i, req in enumerate(requirements):
            tfidf_score = float(cosine_sim[i, j])
            fuzzy_score = int(fuzz.token_set_ratio(req, vtext))
            is_matched = (tfidf_score >= tfidf_th) or (fuzzy_score >= fuzzy_th)
            if is_matched:
                matched_for_vendor.append((i, req, tfidf_score, fuzzy_score))
        matched_counts.append(len(matched_for_vendor))
        matched_for_vendor.sort(key=lambda tup: (tup[2], tup[3]), reverse=True)
        matched_examples.append([t[1] for t in matched_for_vendor[:5]])
        missing = [requirements[i] for i in range(len(requirements)) if all(i != m[0] for m in matched_for_vendor)]
        missing_examples.append(missing[:5])
        matched_matrix.append([1 if any(m[0] == i for m in matched_for_vendor) else 0 for i in range(len(requirements))])

    vendor_df["Matched_Count"] = matched_counts
    vendor_df["Total_Req"] = total_reqs
    vendor_df["Match %"] = vendor_df["Matched_Count"] / vendor_df["Total_Req"] * 100
    if show_examples:
        vendor_df["Matched_Examples"] = [", ".join(x) if x else "" for x in matched_examples]
        vendor_df["Missing_Examples"] = [", ".join(x) if x else "" for x in missing_examples]
    vendor_df["Summary"] = [
        f"Matched {m}/{total_reqs} (examples: {', '.join(ex[:3])})"
        for m, ex in zip(matched_counts, matched_examples)
    ]
    return vendor_df

# Main UI flow
if vendor_file and req_file:
    try:
        vendor_df = pd.read_csv(vendor_file)
        req_df = pd.read_csv(req_file)
    except Exception as e:
        st.error(f"Error reading CSV files: {e}")
        st.stop()

    st.success("Files uploaded successfully!")
    st.write("Detected columns (best-effort):")

    # Guess columns
    vendor_name_col = guess_column(vendor_df, ["name", "vendor", "vendor name", "company"])
    req_title_col = guess_column(req_df, ["essential criteria", "requirement", "criteria", "essential", "title", "requirement title"])
    req_required_col = guess_column(req_df, ["required", "must", "mandatory"])

    # Fallbacks
    if vendor_name_col is None:
        vendor_name_col = vendor_df.columns[0]
    if req_title_col is None:
        # choose first text-like column
        text_cols = [c for c in req_df.columns if req_df[c].dtype == object or req_df[c].dtype == "string"]
        req_title_col = text_cols[0] if text_cols else req_df.columns[0]

    st.write(f"- Vendor name column: **{vendor_name_col}**")
    st.write(f"- Requirement title column: **{req_title_col}**")
    st.write(f"- Required flag column (if any): **{req_required_col}**")

    with st.spinner("Running matching..."):
        results_df = run_matching(vendor_df.copy(), req_df.copy(),
                                  req_title_col=req_title_col,
                                  req_required_col=req_required_col,
                                  tfidf_th=TFIDF_THRESHOLD,
                                  fuzzy_th=FUZZY_THRESHOLD)

    # Prepare shortlist
    shortlist = results_df[results_df["Match %"] >= SHORTLIST_PCT].copy()
    shortlist = shortlist.sort_values("Match %", ascending=False)

    # UI: show top N and overall table
    st.write("### 🏆 Shortlist Results")
    display_cols = [vendor_name_col, "Match %", "Matched_Count", "Summary"]
    if show_examples:
        display_cols += ["Matched_Examples"]
    st.dataframe(shortlist[display_cols].head(50))

    # Download button for shortlist
    csv = shortlist.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Shortlist CSV", csv, "Shortlist_Results.csv", "text/csv")

    # Full results
    with st.expander("Show all vendors and match details"):
        full_display_cols = [vendor_name_col, "Match %", "Matched_Count", "Total_Req", "Summary"]
        if show_examples:
            full_display_cols += ["Matched_Examples", "Missing_Examples"]
        st.dataframe(results_df[full_display_cols].sort_values("Match %", ascending=False).reset_index(drop=True))
        csv_all = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download All Vendors CSV", csv_all, "all_vendors_with_matches.csv", "text/csv")
else:
    st.info("Upload both a Vendor CSV and a Requirements CSV to run matching.")
