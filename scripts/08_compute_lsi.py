"""08_compute_lsi.py - Landslide Susceptibility Index (LSI) Computation (Step 11).

Reclassifies each binned conditioning factor into its Frequency Ratio (FR) weights
using lookup tables and sums the FR weights across all 6 conditioning factors:
    LSI(x, y) = sum_{i=1}^M FR_i(x, y)

Masks non-district cells to NaN and exports:
    - data/processed/final/lsi.tif
    - data/processed/lsi.tif
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import (
    PROC, OUT, MASTER_DIR, FINAL_DIR, TABLES_DIR, FACTORS_BIN_DIR,
    load_grid, load_tif, save_tif
)

FACTORS = [
    ("slope", "cls_slope"),
    ("aspect", "cls_aspect"),
    ("curvature", "cls_curvature"),
    ("dist_roads", "cls_dist_roads"),
    ("dist_streams", "cls_dist_streams"),
    ("landcover", "cls_landcover"),
]


def _load_binned_raster(factor_name, cls_alias):
    """Load binned factor raster trying canonical paths and aliases."""
    candidates = [
        PROC / f"{cls_alias}.tif",
        FACTORS_BIN_DIR / f"{factor_name}.tif",
        FACTORS_BIN_DIR / f"{cls_alias}.tif",
        cls_alias,
        factor_name,
    ]
    for c in candidates:
        try:
            return load_tif(c)
        except Exception:
            continue
    raise FileNotFoundError(f"Could not load binned factor raster for '{factor_name}' (tried {[str(c) for c in candidates]})")


def compute_lsi():
    print("=" * 78)
    print("STEP 11: LANDSLIDE SUSCEPTIBILITY INDEX (LSI) COMPUTATION")
    print("=" * 78)

    # 1. Load master grid definition and district mask
    T, H, W = load_grid()
    mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
    n_valid = int(mask.sum())
    print(f"Master Grid: {H} rows x {W} cols | Valid district cells: {n_valid:,}")

    # 2. Load FR tables
    fr_candidates = [
        OUT / "fr_tables.csv",
        TABLES_DIR / "fr_summary.csv",
        PROC / "fr_tables.csv",
    ]
    fr_path = None
    for p in fr_candidates:
        if p.exists():
            fr_path = p
            break
    if fr_path is None:
        raise FileNotFoundError(f"FR table not found in {[str(c) for c in fr_candidates]}")

    fr_df = pd.read_csv(fr_path)
    print(f"Loaded FR weights from {fr_path}")

    # 3. Sum FR weights across all 6 conditioning factors
    lsi = np.zeros((H, W), dtype="float32")
    factor_summaries = []

    print("\nReclassifying factors with FR lookup tables:")
    for factor_name, cls_layer in FACTORS:
        t = fr_df[fr_df["factor"] == factor_name]
        if t.empty:
            raise ValueError(f"No FR records found for factor '{factor_name}' in {fr_path}")

        cls = _load_binned_raster(factor_name, cls_layer)
        max_cls = int(cls.max())
        lut = np.zeros(max_cls + 1, dtype="float32")

        bins = t["bin"].astype(int).values
        fr_vals = t["FR"].astype(float).values
        lut[bins] = fr_vals

        factor_fr = lut[cls]
        lsi += factor_fr

        f_valid = factor_fr[mask]
        print(f"  - {factor_name:12s} ({cls_layer:18s}): FR min={f_valid.min():.4f}, "
              f"max={f_valid.max():.4f}, mean={f_valid.mean():.4f}")
        factor_summaries.append({
            "factor": factor_name,
            "fr_min": float(f_valid.min()),
            "fr_max": float(f_valid.max()),
            "fr_mean": float(f_valid.mean()),
        })

    # 4. Mask non-district cells to NaN
    lsi[~mask] = np.nan
    valid_lsi = lsi[mask]

    lsi_min = float(np.nanmin(valid_lsi))
    lsi_max = float(np.nanmax(valid_lsi))
    lsi_mean = float(np.nanmean(valid_lsi))
    lsi_std = float(np.nanstd(valid_lsi))
    p5, p25, p50, p75, p95 = [float(x) for x in np.percentile(valid_lsi, [5, 25, 50, 75, 95])]

    print("\nContinuous LSI Distribution Statistics (District Valid Area):")
    print(f"  Count:       {len(valid_lsi):,}")
    print(f"  Min:         {lsi_min:.4f}")
    print(f"  Max:         {lsi_max:.4f}")
    print(f"  Mean:        {lsi_mean:.4f} (Theoretical expectation: 6.000)")
    print(f"  Std Dev:     {lsi_std:.4f}")
    print(f"  Percentiles: 5th={p5:.4f}, 25th={p25:.4f}, Median={p50:.4f}, 75th={p75:.4f}, 95th={p95:.4f}")

    # 5. Export GeoTIFFs
    out_final = FINAL_DIR / "lsi.tif"
    out_proc = PROC / "lsi.tif"

    save_tif(out_final, lsi, T, nodata=np.nan)
    save_tif(out_proc, lsi, T, nodata=np.nan)
    print(f"\nSaved LSI raster to:\n  - {out_final}\n  - {out_proc}")

    return {
        "lsi": lsi,
        "mask": mask,
        "transform": T,
        "stats": {
            "valid_cells": len(valid_lsi),
            "min": lsi_min,
            "max": lsi_max,
            "mean": lsi_mean,
            "std": lsi_std,
            "p5": p5,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
        }
    }


if __name__ == "__main__":
    compute_lsi()
