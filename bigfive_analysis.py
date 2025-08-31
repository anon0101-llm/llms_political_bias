#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Big Five (BFI-50) analysis for GPT-3.5 / GPT-4 / GPT-4o.

- Reads one or more Excel files (local paths OR direct URLs).
- Sheets are roles; columns X1..X50 are Likert items (1..5 or text).
- Computes OCEAN % scores with reverse-keyed items.
- Saves:
    * combined per-response scores (CSV)
    * summary means/SDs by model×role×trait (CSV)
    * pairwise Welch's t-tests + Holm correction + Hedges' g with 95% CI (CSV/XLSX)
    * OCEAN bar chart PDF (default role)
"""

import argparse
import io
import os
import sys
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import requests


# ---------- Configuration ----------

FACTORS = {
    "Openness":         ["X5","X10","X15","X20","X25","X30","X35","X40","X45","X50"],
    "Conscientiousness":["X3","X8","X13","X18","X23","X28","X33","X38","X43","X48"],
    "Extraversion":     ["X1","X6","X11","X16","X21","X26","X31","X36","X41","X46"],
    "Agreeableness":    ["X2","X7","X12","X17","X22","X27","X32","X37","X42","X47"],
    "Neuroticism":      ["X4","X9","X14","X19","X24","X29","X34","X39","X44","X49"],
}

REVERSE = {
    "Openness":         ["X10","X20","X30"],
    "Conscientiousness":["X8","X18","X28","X38"],
    "Extraversion":     ["X6","X16","X26","X36","X46"],
    "Agreeableness":    ["X2","X12","X22","X32"],
    "Neuroticism":      ["X4","X14","X24","X29","X34","X39","X44","X49"],
}

LIKERT_MAP = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}

TRAITS = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
PAIRWISE_MODELS = ["GPT-3.5","GPT-4","GPT-4o"]
ALPHA = 0.05


# ---------- IO helpers ----------

def _read_excel_any(path_or_url: str) -> pd.ExcelFile:
    """Read Excel from local path or HTTP(S) URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.ExcelFile(io.BytesIO(r.content))
    else:
        return pd.ExcelFile(path_or_url)

def _friendly_model_label(basename: str) -> str:
    name = os.path.splitext(os.path.basename(basename.lower()))[0]
    if "gpt3" in name or "gpt3_5" in name or "gpt3-5" in name or "gpt_3_5" in name:
        return "GPT-3.5"
    if "gpt4o" in name or "gpt_4o" in name or "gpt-4o" in name:
        return "GPT-4o"
    if "gpt4" in name or "gpt-4" in name:
        return "GPT-4"
    return name


# ---------- Scoring ----------

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep X* columns, map text Likert to 1..5 if necessary, cast to float."""
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("X")]
    df = df[cols].copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map(LIKERT_MAP)
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

def summarize_sheet(df: pd.DataFrame) -> Dict[str, tuple]:
    """Return dict {trait: (mean, sd)} for OCEAN."""
    out = {}
    for f in TRAITS:
        s = factor_percent_scores(df, f)
        out[f] = (float(s.mean()), float(s.std(ddof=0)))
    return out


# ---------- Stats ----------

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
        t = 0.0 if m1 == m2 else np.nan
        df = np.inf
        p = 1.0 if t == 0.0 else np.nan
        return (m1, s1, n1), (m2, s2, n2), t, df, p
    num = se2**2
    den = ((s1**2)/n1)**2/(n1-1) + ((s2**2)/n2)**2/(n2-1)
    df = np.inf if den == 0 else num/den
    t = (m1 - m2) / np.sqrt(se2)
    p = 2*stats.t.sf(np.abs(t), df)
    return (m1, s1, n1), (m2, s2, n2), t, df, p

def hedges_g_ci(m1, s1, n1, m2, s2, n2, alpha=0.05):
    if any(np.isnan([m1, s1, n1, m2, s2, n2])) or n1 < 2 or n2 < 2:
        return np.nan, (np.nan, np.nan)
    sp2 = (((n1-1)*s1**2) + ((n2-1)*s2**2)) / (n1 + n2 - 2)
    sp2 = max(sp2, 0.0)
    if sp2 == 0:
        d = 0.0 if m1 == m2 else np.nan
    else:
        d = (m1 - m2) / np.sqrt(sp2)
    J = 1 - (3 / (4*(n1+n2) - 9))
    g = J * d if np.isfinite(d) else np.nan
    if not np.isfinite(g):
        return g, (np.nan, np.nan)
    vg = (n1+n2)/(n1*n2) + (g**2)/(2*(n1+n2-2))
    z = stats.norm.ppf(1 - alpha/2)
    lo, hi = g - z*np.sqrt(vg), g + z*np.sqrt(vg)
    return g, (lo, hi)

def fmt_p(p):
    if not np.isfinite(p): return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def fmt_ci(lo, hi):
    return f"[{lo:.2f}, {hi:.2f}]" if np.all(np.isfinite([lo, hi])) else "[n/a]"


# ---------- Plot ----------

def plot_ocean_default(summary_df: pd.DataFrame, out_pdf: str):
    """
    Simple horizontal bar plot of default-role means with SD error bars.
    One bar per trait per model (3 per trait). No seaborn.
    """
    # Keep only Default
    df = summary_df[summary_df["sheet"].str.lower().str.replace("_", "-") == "default"].copy()
    # order traits
    trait_order = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
    models = ["GPT-3.5","GPT-4","GPT-4o"]

    # Make a table with rows = trait, cols = model for means and sds
    means = {m: [df.loc[(df["model"]==m), f"{t}_mean"].mean() for t in trait_order] for m in models}
    sds   = {m: [df.loc[(df["model"]==m), f"{t}_std"].mean()  for t in trait_order] for m in models}

    y = np.arange(len(trait_order))
    h = 0.25

    plt.figure(figsize=(11, 6))
    # Bars: shift by model
    plt.barh(y - h, means["GPT-3.5"], h, xerr=sds["GPT-3.5"], capsize=4, label="GPT-3.5")
    plt.barh(y,       means["GPT-4"],   h, xerr=sds["GPT-4"],   capsize=4, label="GPT-4")
    plt.barh(y + h, means["GPT-4o"],  h, xerr=sds["GPT-4o"],  capsize=4, label="GPT-4o")

    plt.yticks(y, ["O","C","E","A","N"])
    plt.gca().invert_yaxis()
    plt.xlabel("Trait Value [%]")
    plt.ylabel("Big Five Trait")
    plt.xlim(0, 100)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()


# ---------- Main pipeline ----------

def main():
    ap = argparse.ArgumentParser(description="BFI-50 Big Five analysis for GPT models (Welch + Holm).")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Excel files (paths or URLs) for GPT-3.5 / GPT-4 / GPT-4o.")
    ap.add_argument("--outdir", default="./outputs", help="Directory to save outputs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_scores_rows = []
    summary_rows = []

    # Load each Excel, compute per-response scores and per-sheet summaries
    for path in args.inputs:
        try:
            xls = _read_excel_any(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
            continue
        model_label = _friendly_model_label(path)

        for sheet in xls.sheet_names:
            df_raw = pd.read_excel(xls, sheet_name=sheet)
            df_num = to_numeric(df_raw)

            # participant-level percent scores
            scores = {f: factor_percent_scores(df_num, f) for f in TRAITS}
            df_scores = pd.DataFrame(scores)
            df_scores.insert(0, "model", model_label)
            df_scores.insert(1, "sheet", sheet)
            all_scores_rows.append(df_scores)

            # summary
            summ = {"model": model_label, "sheet": sheet, "n": len(df_scores)}
            for f in TRAITS:
                summ[f"{f}_mean"] = float(df_scores[f].mean())
                summ[f"{f}_std"]  = float(df_scores[f].std(ddof=0))
            summary_rows.append(summ)

    if not all_scores_rows:
        print("[ERROR] No data loaded. Check your --inputs paths/URLs.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(all_scores_rows, ignore_index=True)
    summary_df  = pd.DataFrame(summary_rows)

    combined_csv = os.path.join(args.outdir, "combined_bigfive_scores.csv")
    summary_csv  = os.path.join(args.outdir, "summary_bigfive_stats.csv")
    combined_df.to_csv(combined_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Combined scores → {combined_csv}")
    print(f"[OK] Summary stats  → {summary_csv}")

    # Pairwise Welch + Holm + Hedges' g
    results = []
    for role in sorted(combined_df["sheet"].dropna().unique()):
        sub = combined_df[combined_df["sheet"] == role].copy()
        avail = [m for m in PAIRWISE_MODELS if m in sub["model"].unique().tolist()]
        pairs = list(combinations(avail, 2))
        if not pairs:
            continue

        for trait in TRAITS:
            tmp_rows = []
            p_vals = []
            for m1, m2 in pairs:
                a = sub.loc[sub["model"]==m1, trait].dropna().values
                b = sub.loc[sub["model"]==m2, trait].dropna().values
                (mA, sA, nA), (mB, sB, nB), t, df_w, p = welch_t_df(a, b)
                g, (g_lo, g_hi) = hedges_g_ci(mA, sA, nA, mB, sB, nB, alpha=ALPHA)
                tmp_rows.append({
                    "role": role, "trait": trait, "model1": m1, "model2": m2,
                    "n1": int(nA), "n2": int(nB),
                    "mean1": mA, "sd1": sA, "mean2": mB, "sd2": sB,
                    "t": t, "df": df_w, "p_raw": p,
                    "hedges_g": g, "g_ci_low": g_lo, "g_ci_high": g_hi
                })
                p_vals.append(p)

            # Holm correction within (role, trait) across available pairs
            _, p_corr, _, _ = multipletests(p_vals, method="holm")
            # Optional: record step-down thresholds (for reporting)
            m = len(tmp_rows)
            order = np.argsort([r["p_raw"] for r in tmp_rows])
            holm_alpha = [None]*m; stopped = False
            for rank, idx in enumerate(order, start=1):
                thr = ALPHA / (m - rank + 1)
                if (not stopped) and (tmp_rows[idx]["p_raw"] <= thr):
                    holm_alpha[idx] = thr
                else:
                    stopped = True
                    holm_alpha[idx] = None

            for i, row in enumerate(tmp_rows):
                row["p_holm"] = p_corr[i]
                row["holm_alpha"] = holm_alpha[i] if holm_alpha[i] is not None else np.nan
                results.append(row)

    out = pd.DataFrame(results).sort_values(["role","trait","model1","model2"], ignore_index=True)
    out_fmt = out.assign(
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
    )[
        [
            "role","trait","model1","model2","n1","n2",
            "mean1","sd1","mean2","sd2","t","df","p","p_holm","holm_alpha","hedges_g","g_ci"
        ]
    ]

    welch_csv  = os.path.join(args.outdir, "welch_ocean_results.csv")
    welch_xlsx = os.path.join(args.outdir, "welch_ocean_results.xlsx")
    out_fmt.to_csv(welch_csv, index=False)
    try:
        out_fmt.to_excel(welch_xlsx, index=False)
    except Exception:
        pass
    print(f"[OK] Welch/holm table → {welch_csv}")
    print(f"[OK] Welch/holm table (xlsx) → {welch_xlsx}")

    # Plot default means
    plot_path = os.path.join(args.outdir, "OCEAN_BigFive_GPT.pdf")
    plot_ocean_default(summary_df.assign(model=summary_df["model"]), plot_path)
    print(f"[OK] Plot → {plot_path}")

if __name__ == "__main__":
    main()
