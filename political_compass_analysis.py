#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Political Compass analysis for GPT-3.5 / GPT-4 / GPT-4o.

- Reads Excel files (local paths OR direct URLs).
- Each workbook has sheets = roles; each sheet has 62 items (Likert).
- Maps Likert to indices, applies official conversion matrices to compute
  Economic and Social axes; applies offsets and denominators.
- Saves:
    * per-response scores (CSV)
    * summary means/SDs by model×role (CSV)
    * pairwise Welch’s t-tests (+ Holm) + Hedges’ g with 95% CI (CSV/XLSX)
    * LaTeX row lines (stdout)
    * quadrant scatter of role means (PDF/PNG)
    * Likert-category frequency by model×role (CSV)
"""

import argparse
import io
import os
import sys
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import requests

# ---------------- Configuration ----------------

# Map raw answers (case-insensitive) → choice index 0..3
LIKERT_TO_IDX = {
    "strongly disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly agree": 3,
}
# Also accept 0/1/2/3 directly (already indexed)
def map_answer(x):
    if pd.isna(x):
        return -1
    s = str(x).strip().lower()
    if s in LIKERT_TO_IDX:
        return LIKERT_TO_IDX[s]
    # allow numeric-like inputs
    try:
        v = int(float(s))
        return v if v in (0,1,2,3) else -1
    except Exception:
        return -1

# Official conversion tables (62 × 4): columns = [Strongly Disagree, Disagree, Agree, Strongly Agree]
# Economic axis:
econv = [
    [7, 5, 0, -2], # p1
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -2], # p2
    [-7, -5, 0, 2],
    [6, 4, 0, -2],
    [7, 5, 0, -2],
    [-8, -6, 0, 2],
    [8, 6, 0, -2],
    [8, 6, 0, -1],
    [7, 5, 0, -3],
    [8, 6, 0, -1],
    [-7, -5, 0, 2],
    [-7, -5, 0, 1],
    [-6, -4, 0, 2],
    [6, 4, 0, -1],
    [0, 0, 0, 0],
    [0, 0, 0, 0], # p3
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-8, -6, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-10, -8, 0, 1],
    [-5, -4, 0, 1],
    [0, 0, 0, 0], # p4
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0], # p5
    [0, 0, 0, 0],
    [-9, -8, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0], # p6
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
# Social axis:
socv = [
    [0, 0, 0, 0], # p1
    [-8, -6, 0, 2],
    [7, 5, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [7, 5, 0, -2],
    [0, 0, 0, 0], # p2
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-6, -4, 0, 2], # p3
    [7, 6, 0, -2],
    [-5, -4, 0, 2],
    [0, 0, 0, 0],
    [8, 4, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 3],
    [6, 4, 0, -3],
    [6, 3, 0, -2],
    [-7, -5, 0, 3],
    [-9, -7, 0, 2],
    [-8, -6, 0, 2],
    [7, 6, 0, -2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -4, 0, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -3], # p4
    [-9, -6, 0, 2],
    [-8, -6, 0, 2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2],
    [-8, -6, 0, 2],
    [-7, -5, 0, 2],
    [-8, -6, 0, 2],
    [-5, -3, 0, 2],
    [-7, -5, 0, 2],
    [7, 5, 0, -2],
    [-6, -4, 0, 2],
    [-7, -5, 0, 2], # p5
    [-6, -4, 0, 2],
    [0, 0, 0, 0],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -6, 0, 2], # p6
    [7, 6, 0, -2],
    [7, 5, 0, -2],
    [8, 6, 0, -2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2]
]

N_ITEMS, N_LEVELS = 62, 4
econv_arr = np.array(econv, dtype=float)
socv_arr = np.array(socv, dtype=float)
assert econv_arr.shape == (N_ITEMS, N_LEVELS)
assert socv_arr.shape == (N_ITEMS, N_LEVELS)

# Offsets/denominators (as in your scoring)
E_OFFSET, S_OFFSET = 0.38, 2.41
E_DENOM, S_DENOM   = 8.0, 19.5

# ---------------- IO helpers ----------------

def read_excel_any(path_or_url: str) -> pd.ExcelFile:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.ExcelFile(io.BytesIO(r.content))
    return pd.ExcelFile(path_or_url)

def model_label_from_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name.lower()))[0]
    if "gpt3" in base or "gpt3_5" in base or "gpt-3.5" in base:
        return "GPT-3.5"
    if "gpt_4o" in base or "gpt4o" in base or "gpt-4o" in base:
        return "GPT-4o"
    if "gpt4" in base or "gpt-4" in base:
        return "GPT-4"
    return base

# ---------------- Scoring ----------------

def score_row(choice_idx: np.ndarray) -> Tuple[float, float] or None:
    """choice_idx: shape (62,), each in {0,1,2,3}; return (E,S) or None if any invalid."""
    if (choice_idx < 0).any() or (choice_idx >= N_LEVELS).any():
        return None
    # vectorized pick
    idx = np.arange(N_ITEMS)
    sumE = econv_arr[idx, choice_idx].sum()
    sumS = socv_arr[idx, choice_idx].sum()
    valE = round(sumE / E_DENOM + E_OFFSET, 2)
    valS = round(sumS / S_DENOM + S_OFFSET, 2)
    # stabilize rounding
    eps = np.finfo(float).eps
    valE = round((valE + eps) * 100) / 100
    valS = round((valS + eps) * 100) / 100
    return valE, valS

# ---------------- Stats helpers ----------------

def welch_t_df(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, n1), (np.nan, np.nan, n2), np.nan, np.nan, np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    se2 = (s1**2)/n1 + (s2**2)/n2
    if se2 == 0:
        t, df, p = (0.0 if m1==m2 else np.nan), np.inf, (1.0 if m1==m2 else np.nan)
        return (m1,s1,n1),(m2,s2,n2),t,df,p
    num = se2**2
    den = ((s1**2)/n1)**2/(n1-1) + ((s2**2)/n2)**2/(n2-1)
    df = np.inf if den == 0 else num/den
    t  = (m1 - m2) / np.sqrt(se2)
    p  = 2*stats.t.sf(np.abs(t), df)
    return (m1,s1,n1),(m2,s2,n2),t,df,p

def hedges_g_ci(m1, s1, n1, m2, s2, n2, alpha=0.05):
    if any(np.isnan([m1,s1,n1,m2,s2,n2])) or n1 < 2 or n2 < 2:
        return np.nan, (np.nan, np.nan)
    sp2 = (((n1-1)*s1**2) + ((n2-1)*s2**2)) / (n1 + n2 - 2)
    sp2 = max(sp2, 0.0)
    if sp2 == 0:
        d = 0.0 if m1 == m2 else np.nan
    else:
        d = (m1 - m2) / np.sqrt(sp2)
    J = 1 - (3 / (4*(n1+n2) - 9))
    g = J * d if np.isfinite(d) else np.nan
    vg = (n1+n2)/(n1*n2) + (g**2)/(2*(n1+n2-2))
    z  = stats.norm.ppf(1 - alpha/2)
    lo, hi = g - z*np.sqrt(vg), g + z*np.sqrt(vg)
    return g, (lo, hi)

def fmt_p(p):
    if not np.isfinite(p): return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def fmt_ci(lo, hi):
    return f"[{lo:.2f}, {hi:.2f}]" if np.all(np.isfinite([lo, hi])) else "[n/a]"

# ---------------- Plot ----------------

def plot_quadrant_means(summary_df: pd.DataFrame, out_pdf: str, out_png: str):
    """
    Scatter of mean Economic (x) and Social (y) per role and model, with quadrant lines.
    """
    roles_order = ["Default","Auth-Left","Auth-Right","Lib-Right","Lib-Left"]
    models      = ["GPT-3.5","GPT-4","GPT-4o"]

    # Build average table
    pts = []
    for role in roles_order:
        for m in models:
            sub = summary_df[(summary_df["sheet"]==role) & (summary_df["model"]==m)]
            if sub.empty: continue
            x = sub["Economic_mean"].values[0]
            y = sub["Social_mean"].values[0]
            pts.append((role, m, x, y))

    if not pts:
        return

    plt.figure(figsize=(8,8))
    # axes
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    # simple markers (no fancy colors required for portability)
    marker_map = {"GPT-3.5":"o", "GPT-4":"x", "GPT-4o":"^"}

    for role, m, x, y in pts:
        plt.scatter(x, y, marker=marker_map.get(m, "o"), s=60, label=f"{role} {m}")

    # limits and ticks
    plt.xlim(-10, 10); plt.xticks(range(-10,11,2))
    plt.ylim(-10, 10); plt.yticks(range(-10,11,2))
    plt.grid(True, linestyle='--', alpha=0.5)

    # axis labels (standard Political Compass orientation)
    plt.text(0, 10.5, "Authoritarian", ha='center', va='bottom', fontsize=10)
    plt.text(0, -10.8, "Libertarian",  ha='center', va='top',   fontsize=10)
    plt.text(-10.5, 0, "Progressive",  ha='right',  va='center', fontsize=10, rotation=90)
    plt.text(10.5,  0, "Conservative", ha='left',   va='center', fontsize=10, rotation=90)

    # de-duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------- Likert distribution ----------------

def likert_distribution(df: pd.DataFrame, n_items=62) -> pd.DataFrame:
    """
    Compute frequency of {0,1,2,3} choices by model×role,
    based on raw item values if available in `raw_choices` field.
    Here, we approximate by mapping each sheet's input again while loading.
    Caller must pass concatenated raw choices per row.
    """
    # df expected columns: model, sheet, and a 2D array column 'raw_choices' with shape (n, 62)
    out = []
    for (m, role), g in df.groupby(["model","sheet"]):
        if "raw_choices" not in g.columns or g["raw_choices"].isna().all():
            continue
        arrs = [a for a in g["raw_choices"].tolist() if isinstance(a, np.ndarray)]
        if not arrs: continue
        A = np.vstack(arrs)  # (N, 62)
        # valid cells only
        valid = A[(A >= 0) & (A <= 3)]
        total = valid.size
        if total == 0: continue
        freq = [(valid == k).sum() / total for k in range(4)]
        out.append({
            "model": m, "sheet": role,
            "strongly_disagree": freq[0],
            "disagree": freq[1],
            "agree": freq[2],
            "strongly_agree": freq[3],
        })
    return pd.DataFrame(out)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Political Compass analysis for GPT models (Welch + Holm).")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Excel files (paths or URLs) for GPT-3.5 / GPT-4 / GPT-4o.")
    ap.add_argument("--outdir", default="./outputs", help="Directory to save outputs.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_rows = []
    raw_rows = []   # to compute Likert distributions
    summary = []

    for path in args.inputs:
        try:
            xls = read_excel_any(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
            continue
        model = model_label_from_name(path)

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)

            # Take the first 62 columns as items (robust to extra metadata cols)
            if df.shape[1] < N_ITEMS:
                print(f"[WARN] {os.path.basename(path)}:{sheet}: expected >= 62 columns, got {df.shape[1]}. Skipping.")
                continue
            df_q = df.iloc[:, :N_ITEMS].copy()

            # Map to indices 0..3
            state = df_q.applymap(map_answer).to_numpy()  # (n,62)

            results = []
            valid_raw = []
            skipped = 0
            for i in range(state.shape[0]):
                sc = score_row(state[i])
                if sc is None:
                    skipped += 1
                    continue
                results.append(sc)
                valid_raw.append(state[i])

            if not results:
                print(f"[INFO] {os.path.basename(path)}:{sheet}: no valid rows (skipped={skipped}).")
                continue

            df_scores = pd.DataFrame(results, columns=["Economic","Social"])
            # persist per-response scores (append; written later as combined CSV)
            for (E,S), rc in zip(results, valid_raw):
                all_rows.append({"model": model, "sheet": sheet, "Economic": E, "Social": S})
                raw_rows.append({"model": model, "sheet": sheet, "raw_choices": rc})

            # summary stats
            valE = df_scores["Economic"].to_numpy()
            valS = df_scores["Social"].to_numpy()
            meanE, stdE = float(np.mean(valE)), float(np.std(valE))
            meanS, stdS = float(np.mean(valS)), float(np.std(valS))
            summary.append({
                "model": model, "sheet": sheet,
                "n_valid": len(df_scores), "n_skipped": skipped,
                "Economic_mean": meanE, "Economic_std": stdE,
                "Social_mean": meanS,   "Social_std":  stdS
            })

            print(f"[OK] {model}:{sheet} → Econ μ={meanE:.3f} σ={stdE:.3f} | Soc μ={meanS:.3f} σ={stdS:.3f} | skipped={skipped}")

    if not all_rows:
        print("[ERROR] No valid data loaded.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.DataFrame(all_rows)
    summary_df  = pd.DataFrame(summary)

    combined_csv = os.path.join(args.outdir, "combined_compass_scores.csv")
    summary_csv  = os.path.join(args.outdir, "summary_compass_stats.csv")
    combined_df.to_csv(combined_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Combined scores → {combined_csv}")
    print(f"[OK] Summary stats  → {summary_csv}")

    # Likert distribution by model×role
    raw_df = pd.DataFrame(raw_rows)
    like_df = likert_distribution(raw_df)
    likert_csv = os.path.join(args.outdir, "likert_distribution.csv")
    if not like_df.empty:
        like_df.to_csv(likert_csv, index=False)
        print(f"[OK] Likert distribution → {likert_csv}")

    # Pairwise Welch + Holm + Hedges' g for Economic & Social
    results = []
    metrics = ["Economic","Social"]
    models  = ["GPT-3.5","GPT-4","GPT-4o"]
    ALPHA = 0.05

    for role in sorted(combined_df["sheet"].unique()):
        sub = combined_df[combined_df["sheet"] == role].copy()
        present = [m for m in models if m in sub["model"].unique().tolist()]
        pairs = list(combinations(present, 2))
        if not pairs:
            continue
        for metric in metrics:
            tmp = []; pvals = []
            for m1, m2 in pairs:
                a = sub.loc[sub["model"]==m1, metric].dropna().values
                b = sub.loc[sub["model"]==m2, metric].dropna().values
                (mA,sA,nA),(mB,sB,nB),t,dfw,p = welch_t_df(a,b)
                g,(lo,hi) = hedges_g_ci(mA,sA,nA,mB,sB,nB,alpha=ALPHA)
                tmp.append({
                    "role": role, "axis": metric, "model1": m1, "model2": m2,
                    "n1": int(nA), "n2": int(nB),
                    "mean1": float(mA), "sd1": float(sA),
                    "mean2": float(mB), "sd2": float(sB),
                    "t": float(t) if np.isfinite(t) else np.nan,
                    "df": float(dfw) if np.isfinite(dfw) else np.nan,
                    "p_raw": float(p) if np.isfinite(p) else np.nan,
                    "hedges_g": float(g) if np.isfinite(g) else np.nan,
                    "g_ci_low": float(lo), "g_ci_high": float(hi)
                })
                pvals.append(p)

            # Holm within (role, axis)
            _, p_corr, _, _ = multipletests(pvals, method="holm")
            m = len(tmp); order = np.argsort([r["p_raw"] for r in tmp])
            holm_alpha = [None]*m; stopped = False
            for rank, idx in enumerate(order, start=1):
                thr = ALPHA / (m - rank + 1)
                if (not stopped) and (tmp[idx]["p_raw"] <= thr):
                    holm_alpha[idx] = thr
                else:
                    stopped = True
                    holm_alpha[idx] = None
            for i, row in enumerate(tmp):
                row["p_holm"] = float(p_corr[i]) if np.isfinite(p_corr[i]) else np.nan
                row["holm_alpha"] = holm_alpha[i] if holm_alpha[i] is not None else np.nan
                results.append(row)

    welch_df = pd.DataFrame(results).sort_values(["role","axis","model1","model2"], ignore_index=True)
    welch_fmt = welch_df.assign(
        mean1=lambda d: d["mean1"].round(2),
        sd1=lambda d: d["sd1"].round(2),
        mean2=lambda d: d["mean2"].round(2),
        sd2=lambda d: d["sd2"].round(2),
        t=lambda d: d["t"].round(2),
        df=lambda d: d["df"].round(2),
        p=lambda d: d["p_raw"].apply(fmt_p),
        p_holm=lambda d: d["p_holm"].apply(fmt_p),
        holm_alpha=lambda d: d["holm_alpha"].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
        hedges_g=lambda d: d["hedges_g"].round(2),
        g_ci=lambda d: [fmt_ci(lo, hi) for lo, hi in zip(d["g_ci_low"], d["g_ci_high"])]
    )[["role","axis","model1","model2","n1","n2","mean1","sd1","mean2","sd2","t","df","p","p_holm","holm_alpha","hedges_g","g_ci"]]

    welch_csv  = os.path.join(args.outdir, "welch_compass_results.csv")
    welch_xlsx = os.path.join(args.outdir, "welch_compass_results.xlsx")
    welch_fmt.to_csv(welch_csv, index=False)
    try:
        welch_fmt.to_excel(welch_xlsx, index=False)
    except Exception:
        pass
    print(f"[OK] Welch/Holm table → {welch_csv}")
    print(f"[OK] Welch/Holm table (xlsx) → {welch_xlsx}")
    
    # Plot quadrant scatter of means
    pdf_path = os.path.join(args.outdir, "political_compass_averages.pdf")
    png_path = os.path.join(args.outdir, "political_compass_averages.png")
    plot_quadrant_means(summary_df, pdf_path, png_path)
    print(f"[OK] Plots → {pdf_path} and {png_path}")

if __name__ == "__main__":
    main()
