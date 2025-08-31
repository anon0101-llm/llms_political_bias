#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential analysis: Big Five (BFI-50) + Political Compass for LLMs

- Reads one or more Excel workbooks (local paths OR direct URLs).
- Each workbook can be:
    * Two sheets: 'Bigfive' and 'Comprass' (case-insensitive), OR
    * One combined sheet with 112 columns (X1..X112).
        - If sheet name contains 'Bigfive --> Compass' → BF first (X1..X50, X51..X112)
        - If sheet name contains 'Compass --> Bigfive' → PC first (X1..X62, X63..X112)
        - Otherwise auto-detection is attempted.
- Computes:
    * Big Five OCEAN percent scores with reverse-key items
    * Political Compass Economic & Social axes via scoring matrices below
- Merges per-row Big Five and Compass scores (only rows valid in both)
- Saves to --outdir:
    * sequential_combined_scores.csv
    * sequential_summary_stats.csv
    * sequential_correlations_r.csv
    * sequential_correlations_p.csv
"""

import argparse
import io
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr

# ---------- Big Five configuration ----------

FACTORS: Dict[str, List[str]] = {
    "Openness":         ["X5","X10","X15","X20","X25","X30","X35","X40","X45","X50"],
    "Conscientiousness":["X3","X8","X13","X18","X23","X28","X33","X38","X43","X48"],
    "Extraversion":     ["X1","X6","X11","X16","X21","X26","X31","X36","X41","X46"],
    "Agreeableness":    ["X2","X7","X12","X17","X22","X27","X32","X37","X42","X47"],
    "Neuroticism":      ["X4","X9","X14","X19","X24","X29","X34","X39","X44","X49"],
}

REVERSE: Dict[str, List[str]] = {
    "Openness":         ["X10","X20","X30"],
    "Conscientiousness":["X8","X18","X28","X38"],
    "Extraversion":     ["X6","X16","X26","X36","X46"],
    "Agreeableness":    ["X2","X12","X22","X32"],
    "Neuroticism":      ["X4","X14","X24","X29","X34","X39","X44","X49"],
}

# Accept text Likert for Big Five (5-point with Neutral)
LIKERT_MAP_BF = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}

TRAITS = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

# ---------- Political Compass configuration ----------

# Map raw answers (case-insensitive) → choice index 0..3 (SD, D, A, SA)
LIKERT_TO_IDX_PC = {
    "strongly disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly agree": 3,
}

def map_compass_answer(x):
    """Robust mapping for Political Compass cells to {0,1,2,3}, else -1."""
    if pd.isna(x):
        return -1
    s = str(x).strip().lower()
    if s in LIKERT_TO_IDX_PC:
        return LIKERT_TO_IDX_PC[s]
    # allow numeric encodings 0..3
    try:
        v = int(float(s))
        return v if v in (0,1,2,3) else -1
    except Exception:
        return -1

# Economic and Social scoring matrices (62 × 4)
econv = [
    [7, 5, 0, -2], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [7, 5, 0, -2], [-7,-5,0, 2], [6, 4,0,-2], [7, 5,0,-2], [-8,-6,0, 2],
    [8, 6, 0,-2], [8, 6,0,-1], [7, 5,0,-3], [8, 6,0,-1], [-7,-5,0, 2], [-7,-5,0, 1],
    [-6,-4, 0, 2], [6, 4,0,-1], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [-8,-6,0, 1], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0], [-10,-8,0,1], [-5,-4,0,1], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [-9,-8,0,1], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0]
]
socv = [
    [0,0,0,0], [-8,-6,0,2], [7,5,0,-2], [-7,-5,0,2], [-7,-5,0,2], [-6,-4,0,2], [7,5,0,-2],
    [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0],
    [-6,-4,0,2], [7,6,0,-2], [-5,-4,0,2], [0,0,0,0], [8,4,0,-2], [-7,-5,0,2],
    [-7,-5,0,3], [6,4,0,-3], [6,3,0,-2], [-7,-5,0,3], [-9,-7,0,2], [-8,-6,0,2],
    [7,6,0,-2], [-7,-5,0,2], [-6,-4,0,2], [-7,-4,0,2], [0,0,0,0], [0,0,0,0],
    [7,5,0,-3], [-9,-6,0,2], [-8,-6,0,2], [-8,-6,0,2], [-6,-4,0,2], [-8,-6,0,2],
    [-7,-5,0,2], [-8,-6,0,2], [-5,-3,0,2], [-7,-5,0,2], [7,5,0,-2], [-6,-4,0,2],
    [-7,-5,0,2], [-6,-4,0,2], [0,0,0,0], [-7,-5,0,2], [-6,-4,0,2], [-7,-6,0,2],
    [7,6,0,-2], [7,5,0,-2], [8,6,0,-2], [-8,-6,0,2], [-6,-4,0,2]
]

N_ITEMS, N_LEVELS = 62, 4
econv_arr = np.array(econv, dtype=float)
socv_arr = np.array(socv, dtype=float)
assert econv_arr.shape == (N_ITEMS, N_LEVELS)
assert socv_arr.shape == (N_ITEMS, N_LEVELS)

E_OFFSET, S_OFFSET = 0.38, 2.41
E_DENOM, S_DENOM   = 8.0, 19.5

# ---------- IO helpers ----------

def read_excel_any(path_or_url: str) -> pd.ExcelFile:
    """Read Excel from local path or HTTP(S) URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.ExcelFile(io.BytesIO(r.content))
    return pd.ExcelFile(path_or_url)

def model_label_from_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name.lower()))[0]
    if "gpt3" in base or "gpt3_5" in base or "gpt-3.5" in base or "gpt_3_5" in base:
        return "GPT-3.5"
    if "gpt_4o" in base or "gpt4o" in base or "gpt-4o" in base:
        return "GPT-4o"
    if "gpt4" in base or "gpt-4" in base:
        return "GPT-4"
    return base

def order_hint_from_sheet_name(sheet_name: str):
    """
    Infer order from sheet name.
    Expected patterns (case-insensitive):
      - 'Bigfive --> Compass'  → BF first (50 then 62)
      - 'Compass --> Bigfive'  → PC first (62 then 50)
    Falls back to relative position of tokens if '–>' missing; else None.
    """
    s = (sheet_name or "").lower().strip()
    s = s.replace("->", "-->").replace("—>", "-->").replace("→", "-->")
    big_idx = s.find("big")
    comp_idx = s.find("comp")
    arrow = s.find("-->")
    if big_idx != -1 and comp_idx != -1 and arrow != -1:
        left = s[:arrow].strip()
        right = s[arrow+3:].strip()
        if "big" in left and "comp" in right:
            return "BF_FIRST"
        if "comp" in left and "big" in right:
            return "PC_FIRST"
    if big_idx != -1 and comp_idx != -1:
        return "BF_FIRST" if big_idx < comp_idx else "PC_FIRST"
    return None

# ---------- Big Five scoring ----------

def to_numeric_bigfive(df: pd.DataFrame) -> pd.DataFrame:
    """Keep X* columns, map text Likert to 1..5 if necessary, cast to float."""
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("X")]
    df = df[cols].copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map(LIKERT_MAP_BF)
    return df.astype(float)

def reverse_score(df: pd.DataFrame, items: List[str]) -> pd.DataFrame:
    """Reverse score specified items in-place: 6 - x."""
    for it in items:
        if it in df.columns:
            df[it] = 6 - df[it]
    return df

def factor_percent_scores(df: pd.DataFrame, factor: str) -> pd.Series:
    """Mean of 10 items (after reverse where needed) → %: (mean-1)/4*100."""
    items = FACTORS[factor]
    sub = df[items].copy()
    sub = reverse_score(sub, REVERSE.get(factor, []))
    mean_row = sub.mean(axis=1)
    return (mean_row - 1) / 4 * 100

def score_bigfive_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns O,C,E,A,N (%), preserving original row order index."""
    df_num = to_numeric_bigfive(df_raw)
    out = pd.DataFrame({
        "Openness":          factor_percent_scores(df_num, "Openness"),
        "Conscientiousness": factor_percent_scores(df_num, "Conscientiousness"),
        "Extraversion":      factor_percent_scores(df_num, "Extraversion"),
        "Agreeableness":     factor_percent_scores(df_num, "Agreeableness"),
        "Neuroticism":       factor_percent_scores(df_num, "Neuroticism"),
    }, index=df_num.index)
    return out

# ---------- Political Compass scoring ----------

def score_compass_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map first 62 columns to indices 0..3, skip rows with any invalid cell,
    and compute Economic/Social scores.
    """
    if df_raw.shape[1] < N_ITEMS:
        raise ValueError(f"Expected >= {N_ITEMS} columns, got {df_raw.shape[1]}")
    df_q = df_raw.iloc[:, :N_ITEMS].copy()
    state = df_q.applymap(map_compass_answer).to_numpy()  # (n,62)

    idx = np.arange(N_ITEMS)
    econ_vals, soc_vals, keep_idx = [], [], []
    for i in range(state.shape[0]):
        row = state[i]
        if (row < 0).any():
            continue  # skip invalid row
        sumE = econv_arr[idx, row].sum()
        sumS = socv_arr[idx, row].sum()
        valE = round(sumE / E_DENOM + E_OFFSET, 2)
        valS = round(sumS / S_DENOM + S_OFFSET, 2)
        # stabilize rounding
        eps = np.finfo(float).eps
        valE = round((valE + eps) * 100) / 100
        valS = round((valS + eps) * 100) / 100
        econ_vals.append(valE)
        soc_vals.append(valS)
        keep_idx.append(i)

    out = pd.DataFrame({"Economic": econ_vals, "Social": soc_vals}, index=keep_idx)
    return out

# ---------- Splitter for single-sheet X1..X112 ----------

BF_TOKENS = {"strongly disagree","disagree","neutral","agree","strongly agree"}
PC_TOKENS = {"strongly disagree","disagree","agree","strongly agree"}  # no neutral

def _valid_ratio_bigfive(df: pd.DataFrame) -> float:
    """Fraction of cells compatible with 5-point Likert including 'neutral'."""
    vals = df.astype(str).str.strip().str.lower()
    return (vals.isin(BF_TOKENS)).mean().mean()

def _valid_ratio_compass(df: pd.DataFrame) -> float:
    """Fraction of cells compatible with 4-point Likert (words) OR integers 0..3."""
    vals = df.astype(str).str.strip().str.lower()
    ok_words = vals.isin(PC_TOKENS)
    ok_ints  = vals.applymap(lambda s: s.isdigit() and int(s) in (0,1,2,3))
    return ((ok_words | ok_ints)).mean().mean()

def split_combined_112(df_raw: pd.DataFrame, order_hint: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single 112-column sheet (X1..X112) into (df_bf, df_pc).
    If order_hint is:
        - 'BF_FIRST' → BF = X1..X50, PC = X51..X112
        - 'PC_FIRST' → PC = X1..X62, BF = X63..X112
    Otherwise falls back to auto-detect by validity ratios.
    """
    cols = [c for c in df_raw.columns if isinstance(c, str) and c.startswith("X")]
    if len(cols) < 112:
        raise ValueError(f"Expected 112 X-columns, found {len(cols)}")
    cols_sorted = sorted(cols, key=lambda s: int(s[1:]))  # X1..X112
    df = df_raw[cols_sorted].copy()

    # honor hint if present
    if order_hint == "BF_FIRST":
        return df.iloc[:, 0:50], df.iloc[:, 50:112]
    if order_hint == "PC_FIRST":
        return df.iloc[:, 62:112], df.iloc[:, 0:62]

    # fallback: auto-detect
    A_bf = df.iloc[:, 0:50]
    A_pc = df.iloc[:, 50:112]
    A_score = _valid_ratio_bigfive(A_bf) + _valid_ratio_compass(A_pc)

    B_pc = df.iloc[:, 0:62]
    B_bf = df.iloc[:, 62:112]
    B_score = _valid_ratio_compass(B_pc) + _valid_ratio_bigfive(B_bf)

    return (A_bf, A_pc) if A_score >= B_score else (B_bf, B_pc)

# ---------- Correlation helpers ----------

ALL_COLS = ["Economic","Social","Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

def pearson_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (r-matrix, p-matrix) with pairwise Pearson correlations over rows with non-NaN pairs."""
    r = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            x = df[i]; y = df[j]
            m = x.notna() & y.notna()
            if m.sum() >= 3:
                rr, pp = pearsonr(x[m].to_numpy(), y[m].to_numpy())
                r.loc[i,j] = rr
                p.loc[i,j] = pp
            else:
                r.loc[i,j] = np.nan
                p.loc[i,j] = np.nan
    return r, p

# ---------- Main ----------

def read_excel_any(path_or_url: str) -> pd.ExcelFile:
    """(Re-declared above intentionally)"""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.ExcelFile(io.BytesIO(r.content))
    return pd.ExcelFile(path_or_url)

def main():
    ap = argparse.ArgumentParser(description="Sequential Big Five + Political Compass analysis.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Excel files (paths or URLs). Accepts (A) 'Bigfive' + 'Comprass' sheets, or (B) a single sheet with X1..X112.")
    ap.add_argument("--outdir", default="./outputs", help="Directory to save outputs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    combined_rows = []
    summaries = []

    for path in args.inputs:
        try:
            xls = read_excel_any(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
            continue

        model = model_label_from_name(path)
        file_tag = os.path.basename(path)

        # Try legacy 2-sheet layout first
        sheets_lower = {s.lower(): s for s in xls.sheet_names}
        s_big = sheets_lower.get("bigfive")
        s_cmp = sheets_lower.get("comprass")

        try:
            if s_big and s_cmp:
                df_bf_raw = pd.read_excel(xls, sheet_name=s_big)
                df_cp_raw = pd.read_excel(xls, sheet_name=s_cmp)
            else:
                # Fallback: single-sheet combined format
                if len(xls.sheet_names) != 1:
                    print(f"[WARN] {file_tag}: expected 1 combined sheet or named Bigfive/Comprass; found {len(xls.sheet_names)}. Trying first sheet.", file=sys.stderr)
                sheet0 = xls.sheet_names[0]
                df0 = pd.read_excel(xls, sheet_name=sheet0)
                hint = order_hint_from_sheet_name(sheet0)
                df_bf_raw, df_cp_raw = split_combined_112(df0, order_hint=hint)
        except Exception as e:
            print(f"[ERROR] {file_tag}: sheet loading/splitting failed: {e}", file=sys.stderr)
            continue

        # Score both parts
        try:
            df_bf = score_bigfive_sheet(df_bf_raw)
        except Exception as e:
            print(f"[WARN] {file_tag}: Bigfive scoring failed: {e}", file=sys.stderr)
            continue

        try:
            df_cp = score_compass_sheet(df_cp_raw)
        except Exception as e:
            print(f"[WARN] {file_tag}: Compass scoring failed: {e}", file=sys.stderr)
            continue

        # Merge on row index (keep only rows valid in both)
        df_merged = df_cp.join(df_bf, how="inner")
        if df_merged.empty:
            print(f"[INFO] {file_tag}: no overlapping valid rows. Skipping merge.", file=sys.stderr)
            continue

        # Attach tags and row ids
        df_merged = df_merged.reset_index(drop=True)
        df_merged.insert(0, "row_id", df_merged.index + 1)
        df_merged.insert(0, "file", file_tag)
        df_merged.insert(0, "model", model)
        combined_rows.append(df_merged)

        # Simple per-file summary
        summ = {"file": file_tag, "model": model, "n": len(df_merged)}
        for col in ["Economic","Social"] + TRAITS:
            summ[f"{col}_mean"] = float(df_merged[col].mean())
            summ[f"{col}_std"]  = float(df_merged[col].std(ddof=0))
        summaries.append(summ)

        print(f"[OK] {file_tag}: merged rows = {len(df_merged)}")

    if not combined_rows:
        print("[ERROR] No data merged across inputs.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    # Save combined scores and per-file summary
    combined_csv = os.path.join(args.outdir, "sequential_combined_scores.csv")
    summary_csv  = os.path.join(args.outdir, "sequential_summary_stats.csv")
    combined_df.to_csv(combined_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    print(f"[OK] Combined scores  → {combined_csv}")
    print(f"[OK] Summary stats    → {summary_csv}")

    # Overall correlations across all merged rows
    r_mat, p_mat = pearson_matrix(combined_df, ALL_COLS)
    r_csv = os.path.join(args.outdir, "sequential_correlations_r.csv")
    p_csv = os.path.join(args.outdir, "sequential_correlations_p.csv")
    r_mat.to_csv(r_csv)
    p_mat.to_csv(p_csv)
    print(f"[OK] Correlation (r)  → {r_csv}")
    print(f"[OK] Correlation (p)  → {p_csv}")

if __name__ == "__main__":
    main()
