"""07_compute_frequency_ratio.py - Vectorized Frequency Ratio (FR) Computation (Step 10).

Calculates the Frequency Ratio (FR) for each conditioning factor class:
  FR = (% training landslides in class) / (% district area in class)
     = (N_ij / N_total) / (A_ij / A_total)

Vectorized implementation using np.bincount over training landslide pixels.
Generates and exports:
  - reports/tables/fr_summary.csv
  - reports/tables/fr_summary.md
  - reports/tables/fr_summary.tex
  - outputs/fr_tables.csv
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import (
    PROC, OUT, REPORTS, TABLES_DIR, SAMPLES_DIR,
    load_tif
)

# Factor human-readable metadata & class labels
FACTOR_LABELS = {
    "slope": {
        1: "0-10 deg (Gentle)",
        2: "10-20 deg (Moderate)",
        3: "20-30 deg (Moderately steep)",
        4: "30-40 deg (Steep)",
        5: ">40 deg (Very steep)",
    },
    "aspect": {
        1: "North (337.5-22.5 deg)",
        2: "Northeast (22.5-67.5 deg)",
        3: "East (67.5-112.5 deg)",
        4: "Southeast (112.5-157.5 deg)",
        5: "South (157.5-202.5 deg)",
        6: "Southwest (202.5-247.5 deg)",
        7: "West (247.5-292.5 deg)",
        8: "Northwest (292.5-337.5 deg)",
        9: "Flat (Slope < 1 deg)",
    },
    "curvature": {
        1: "Concave (<-0.005)",
        2: "Flat (-0.005 to +0.005)",
        3: "Convex (>+0.005)",
    },
    "dist_roads": {
        1: "0-50 m",
        2: "50-100 m",
        3: "100-200 m",
        4: "200-300 m",
        5: ">300 m",
    },
    "dist_streams": {
        1: "0-50 m",
        2: "50-100 m",
        3: "100-200 m",
        4: "200-300 m",
        5: ">300 m",
    },
    "landcover": {
        1: "Tree cover",
        2: "Shrubland",
        3: "Grassland",
        4: "Cropland",
        5: "Built-up",
        6: "Bare / sparse vegetation",
        7: "Permanent water bodies",
        8: "Herbaceous wetland",
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
    },
}


def compute_fr_table(cls, train_rc, factor_name):
    """Vectorized calculation of area, training hits, percentages, and FR for a factor."""
    n = int(cls.max()) + 1
    area = np.bincount(cls[cls > 0], minlength=n)
    hits = np.bincount(cls[train_rc[:, 0], train_rc[:, 1]], minlength=n)
    
    df = pd.DataFrame({"class_id": np.arange(n), "pixels": area, "slides": hits})
    df = df[(df.class_id > 0) & (df.pixels > 0)].copy()
    
    total_pixels = df.pixels.sum()
    total_slides = df.slides.sum()
    
    df["area_pct"] = 100.0 * df.pixels / total_pixels
    df["slide_pct"] = 100.0 * df.slides / total_slides
    df["FR"] = df.slide_pct / df.area_pct
    df["bin"] = df["class_id"]  # Ensure Step 11 guide compatibility: lut[t['bin']]
    
    # Add human readable factor names and class descriptions
    labels_dict = FACTOR_LABELS.get(factor_name, {})
    df["class_label"] = df["class_id"].map(lambda cid: labels_dict.get(int(cid), f"Class {cid}"))
    df.insert(0, "factor", factor_name)
    
    # Validate conservation
    assert abs(df.area_pct.sum() - 100.0) < 1e-5, f"Area % does not sum to 100 for {factor_name}"
    assert abs(df.slide_pct.sum() - 100.0) < 1e-5, f"Slide % does not sum to 100 for {factor_name}"
    return df


def generate_fr_summary():
    print("=" * 78)
    print("STEP 10: VECTORIZED FREQUENCY RATIO (FR) COMPUTATION")
    print("=" * 78)

    # 1. Load training points
    npz_candidates = [
        SAMPLES_DIR / "points.npz",
        PROC / "points.npz",
    ]
    npz_path = None
    for p in npz_candidates:
        if p.exists():
            npz_path = p
            break
    if npz_path is None:
        raise FileNotFoundError(f"Training points NPZ not found in {[str(c) for c in npz_candidates]}")

    pts = np.load(npz_path)
    train_rc = pts["train"]
    print(f"Loaded {len(train_rc)} training landslide coordinates from {npz_path}")

    # Factors to evaluate
    factors = [
        ("slope", "cls_slope"),
        ("aspect", "cls_aspect"),
        ("curvature", "cls_curvature"),
        ("dist_roads", "cls_dist_roads"),
        ("dist_streams", "cls_dist_streams"),
        ("landcover", "cls_landcover"),
    ]

    tables = []
    print("\nComputing Frequency Ratio tables per factor...")
    for f_name, cls_layer in factors:
        cls_arr = load_tif(cls_layer)
        t = compute_fr_table(cls_arr, train_rc, f_name)
        tables.append(t)
        print(f"  Processed {f_name:<14} -> {len(t)} classes, slides={t.slides.sum()}, area_pct_sum={t.area_pct.sum():.2f}%")

    fr_all = pd.concat(tables, ignore_index=True)

    # Export CSVs
    csv_out_path = TABLES_DIR / "fr_summary.csv"
    fr_all.to_csv(csv_out_path, index=False)
    fr_all.to_csv(OUT / "fr_tables.csv", index=False)
    fr_all.to_csv(PROC / "fr_tables.csv", index=False)
    print(f"\nSaved consolidated FR CSV to {csv_out_path} and {OUT / 'fr_tables.csv'}")

    # Export Markdown table
    md_out_path = TABLES_DIR / "fr_summary.md"
    md_lines = [
        "# Frequency Ratio (FR) Analysis Summary - Wayanad District",
        "",
        f"Computed from {len(train_rc)} training landslides on 30m master grid.",
        "",
        "| Factor | Class ID | Class Label | District Pixels | Area % | Training Slides | Slides % | Frequency Ratio (FR) | Susceptibility Indication |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, r in fr_all.iterrows():
        fr_val = r["FR"]
        if fr_val > 1.5:
            indication = "High Susceptibility (FR > 1.5)"
        elif fr_val > 1.0:
            indication = "Moderate Susceptibility (FR > 1.0)"
        elif fr_val == 0.0:
            indication = "Zero Slides (Stable / Gap)"
        else:
            indication = "Low Susceptibility (FR < 1.0)"
            
        md_lines.append(
            f"| `{r['factor']}` | {int(r['class_id'])} | {r['class_label']} | "
            f"{int(r['pixels']):,d} | {r['area_pct']:.2f}% | {int(r['slides']):d} | "
            f"{r['slide_pct']:.2f}% | **{r['FR']:.3f}** | {indication} |"
        )
    md_out_path.write_text("\n".join(md_lines))
    print(f"Saved Markdown summary table to {md_out_path}")

    # Export LaTeX table
    tex_out_path = TABLES_DIR / "fr_summary.tex"
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Frequency Ratio (FR) Values for Landslide Conditioning Factors in Wayanad District}",
        r"\label{tab:fr_summary}",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"\textbf{Factor} & \textbf{Class Label} & \textbf{Pixels} & \textbf{Area \%} & \textbf{Slides} & \textbf{Slide \%} & \textbf{FR} \\",
        r"\hline",
    ]
    curr_factor = None
    for _, r in fr_all.iterrows():
        f = r["factor"]
        f_display = f.replace("_", r"\_") if f != curr_factor else ""
        curr_factor = f
        label_display = str(r["class_label"]).replace("<", r"$<$").replace(">", r"$>$").replace("_", r"\_").replace("%", r"\%")
        tex_lines.append(
            f"{f_display} & {label_display} & {int(r['pixels']):,d} & {r['area_pct']:.2f} & {int(r['slides'])} & {r['slide_pct']:.2f} & \\textbf{{{r['FR']:.3f}}} \\\\"
        )
    tex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_out_path.write_text("\n".join(tex_lines))
    print(f"Saved LaTeX summary table to {tex_out_path}")

    # Terminal presentation table
    print("\n" + "=" * 92)
    print("FREQUENCY RATIO (FR) RESULTS TABLE")
    print("=" * 92)
    print(f"{'Factor':<13} {'Class':<4} {'Label':<28} {'Area Pixels':<13} {'Area %':<9} {'Slides':<7} {'Slide %':<9} {'FR':<7}")
    print("-" * 92)
    for _, r in fr_all.iterrows():
        print(
            f"{r['factor']:<13} {int(r['class_id']):<4d} {r['class_label'][:27]:<28} "
            f"{int(r['pixels']):<13,d} {r['area_pct']:<9.2f} {int(r['slides']):<7d} "
            f"{r['slide_pct']:<9.2f} {r['FR']:<7.3f}"
        )
    print("=" * 92)
    print("ALL FREQUENCY RATIO COMPUTATIONS COMPLETED SUCCESSFULLY!\n")

    return fr_all


if __name__ == "__main__":
    generate_fr_summary()
