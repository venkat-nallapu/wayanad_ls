"""05_bin_conditioning_factors.py - Discretization of Conditioning Factors (Step 8).

Discretizes continuous conditioning factors into integer classes using np.digitize
with the geomorphological bins specified in Section 8 of the Project Guide:
1. Slope (5 classes): 1: 0–10°, 2: 10–20°, 3: 20–30°, 4: 30–40°, 5: >40°
2. Curvature (3 classes): 1: Concave (< -0.005), 2: Flat (-0.005 to +0.005), 3: Convex (> +0.005)
3. Distance to Roads (5 classes): 1: 0–50m, 2: 50–100m, 3: 100–200m, 4: 200–300m, 5: >300m
4. Distance to Streams (5 classes): 1: 0–50m, 2: 50–100m, 3: 100–200m, 4: 200–300m, 5: >300m
5. Aspect (9 classes): 1..8 compass directions (N, NE, E, SE, S, SW, W, NW) + 9: Flat
6. Land Cover: ESA WorldCover classes mapped with contiguous class IDs (1..K)
   and dictionary mapping saved to JSON.

All binned rasters are uint8 with 0 = NoData (masked outside district boundary).
Outputs saved to:
  - data/processed/factors_binned/
  - data/processed/ (cls_* guide aliases)
  - outputs/ & reports/tables/ (bin_edges.csv)
"""
import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import (
    PROC, OUT, REPORTS, TABLES_DIR, MASTER_DIR, FACTORS_CONT_DIR, FACTORS_BIN_DIR,
    load_grid, load_tif, save_tif
)

# Geomorphological bin definitions
BIN_DEFINITIONS = {
    "slope": {
        "edges": [10.0, 20.0, 30.0, 40.0],
        "labels": {
            1: "0-10 deg (Gentle)",
            2: "10-20 deg (Moderate)",
            3: "20-30 deg (Moderately steep)",
            4: "30-40 deg (Steep)",
            5: ">40 deg (Very steep)",
        },
        "description": "Slope angle in degrees",
    },
    "curvature": {
        "edges": [-0.005, 0.005],
        "labels": {
            1: "Concave (<-0.005)",
            2: "Flat (-0.005 to +0.005)",
            3: "Convex (>+0.005)",
        },
        "description": "Profile/plan curvature (Laplacian of elevation)",
    },
    "dist_roads": {
        "edges": [50.0, 100.0, 200.0, 300.0],
        "labels": {
            1: "0-50 m",
            2: "50-100 m",
            3: "100-200 m",
            4: "200-300 m",
            5: ">300 m",
        },
        "description": "Euclidean distance to nearest road in meters",
    },
    "dist_streams": {
        "edges": [50.0, 100.0, 200.0, 300.0],
        "labels": {
            1: "0-50 m",
            2: "50-100 m",
            3: "100-200 m",
            4: "200-300 m",
            5: ">300 m",
        },
        "description": "Euclidean distance to nearest stream in meters",
    },
    "aspect": {
        "edges": None,  # Pre-computed compass classes 1..9
        "labels": {
            1: "North (337.5 - 22.5 deg)",
            2: "Northeast (22.5 - 67.5 deg)",
            3: "East (67.5 - 112.5 deg)",
            4: "Southeast (112.5 - 157.5 deg)",
            5: "South (157.5 - 202.5 deg)",
            6: "Southwest (202.5 - 247.5 deg)",
            7: "West (247.5 - 292.5 deg)",
            8: "Northwest (292.5 - 337.5 deg)",
            9: "Flat (Slope < 1 deg)",
        },
        "description": "Downslope compass aspect bearing",
    },
}

# ESA WorldCover official code definitions & contiguous mapping
ESA_LEGEND = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
}

ESA_TO_CONTIGUOUS = {
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    50: 5,
    60: 6,
    80: 7,
    90: 8,
}

CONTIGUOUS_TO_ESA = {v: k for k, v in ESA_TO_CONTIGUOUS.items()}
CONTIGUOUS_NAMES = {v: ESA_LEGEND[k] for k, v in ESA_TO_CONTIGUOUS.items()}


def digitize_factor(arr, edges, valid_mask):
    """Discretize continuous raster into 1-based integer classes using np.digitize.
    
    0 = nodata (outside valid_mask or non-finite values).
    Classes: 1 .. len(edges) + 1.
    """
    arr = np.asarray(arr, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    cls = np.digitize(arr, edges) + 1
    cls = np.where(valid_mask & np.isfinite(arr), cls, 0).astype("uint8")
    return cls


def bin_conditioning_factors():
    T, H, W = load_grid()
    mask_path = MASTER_DIR / "mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"District mask not found at {mask_path}")
    mask = load_tif(mask_path).astype(bool)
    n_dist = int(mask.sum())
    print(f"Loaded master grid ({H} x {W}) with {n_dist:,} valid district cells.")

    binned_layers = {}

    # 1. Slope (5 bins)
    print("\n[1/6] Discretizing Slope (5 classes: 0-10, 10-20, 20-30, 30-40, >40 deg)...")
    slope_raw = load_tif(FACTORS_CONT_DIR / "slope.tif")
    binned_layers["slope"] = digitize_factor(slope_raw, BIN_DEFINITIONS["slope"]["edges"], mask)

    # 2. Curvature (3 bins)
    print("[2/6] Discretizing Curvature (3 classes: Concave <-0.005, Flat -0.005..0.005, Convex >0.005)...")
    curv_raw = load_tif(FACTORS_CONT_DIR / "curv.tif")
    binned_layers["curvature"] = digitize_factor(curv_raw, BIN_DEFINITIONS["curvature"]["edges"], mask)

    # 3. Distance to Roads (5 bins)
    print("[3/6] Discretizing Distance to Roads (5 classes: 0-50, 50-100, 100-200, 200-300, >300 m)...")
    d_road_raw = load_tif(FACTORS_CONT_DIR / "dist_roads.tif")
    binned_layers["dist_roads"] = digitize_factor(d_road_raw, BIN_DEFINITIONS["dist_roads"]["edges"], mask)

    # 4. Distance to Streams (5 bins)
    print("[4/6] Discretizing Distance to Streams (5 classes: 0-50, 50-100, 100-200, 200-300, >300 m)...")
    d_stream_raw = load_tif(FACTORS_CONT_DIR / "dist_streams.tif")
    binned_layers["dist_streams"] = digitize_factor(d_stream_raw, BIN_DEFINITIONS["dist_streams"]["edges"], mask)

    # 5. Aspect (9 classes: already binned in Step 2)
    print("[5/6] Loading Aspect (9 compass classes)...")
    aspect_binned = load_tif(FACTORS_BIN_DIR / "aspect.tif")
    aspect_binned = np.where(mask, aspect_binned, 0).astype("uint8")
    binned_layers["aspect"] = aspect_binned

    # 6. Land Cover (ESA WorldCover classes and contiguous 1..K mapping)
    print("[6/6] Mapping Land Cover codes (contiguous 1..K and ESA standard)...")
    lc_raw = load_tif(FACTORS_BIN_DIR / "landcover.tif")
    
    # Check if lc_raw is already ESA codes or contiguous
    if set(np.unique(lc_raw[mask])).issubset(set(ESA_LEGEND.keys())):
        lc_esa = lc_raw.copy()
        lc_contiguous = np.zeros_like(lc_raw, dtype="uint8")
        for esa_code, c_id in ESA_TO_CONTIGUOUS.items():
            lc_contiguous[lc_raw == esa_code] = c_id
    else:
        # If already contiguous, map back
        lc_contiguous = lc_raw.copy()
        lc_esa = np.zeros_like(lc_raw, dtype="uint8")
        for c_id, esa_code in CONTIGUOUS_TO_ESA.items():
            lc_esa[lc_raw == c_id] = esa_code

    lc_contiguous = np.where(mask, lc_contiguous, 0).astype("uint8")
    lc_esa = np.where(mask, lc_esa, 0).astype("uint8")
    binned_layers["landcover"] = lc_contiguous
    binned_layers["landcover_contiguous"] = lc_contiguous
    binned_layers["landcover_esa"] = lc_esa

    # Save landcover label mapping dictionary to JSON
    mapping_payload = {
        "esa_to_contiguous": {str(k): v for k, v in ESA_TO_CONTIGUOUS.items()},
        "contiguous_to_esa": {str(k): v for k, v in CONTIGUOUS_TO_ESA.items()},
        "contiguous_to_name": {str(k): v for k, v in CONTIGUOUS_NAMES.items()},
        "esa_legend": {str(k): v for k, v in ESA_LEGEND.items()},
    }
    legend_json_path = FACTORS_BIN_DIR / "landcover_legend.json"
    legend_json_path.write_text(json.dumps(mapping_payload, indent=2))
    (FACTORS_BIN_DIR / "landcover_mapping.json").write_text(json.dumps(mapping_payload, indent=2))
    print(f"Saved land cover legend mapping to {legend_json_path}")

    # Save GeoTIFFs to data/processed/factors_binned/
    print("\nSaving binned factor GeoTIFFs to data/processed/factors_binned/...")
    save_tif(FACTORS_BIN_DIR / "slope.tif", binned_layers["slope"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "curvature.tif", binned_layers["curvature"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "curv.tif", binned_layers["curvature"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "dist_roads.tif", binned_layers["dist_roads"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "d_road.tif", binned_layers["dist_roads"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "dist_streams.tif", binned_layers["dist_streams"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "d_stream.tif", binned_layers["dist_streams"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "aspect.tif", binned_layers["aspect"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "aspect_binned.tif", binned_layers["aspect"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "landcover.tif", binned_layers["landcover"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "landcover_contiguous.tif", binned_layers["landcover_contiguous"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "landcover_binned.tif", binned_layers["landcover_contiguous"], T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "landcover_esa.tif", binned_layers["landcover_esa"], T, nodata=0)

    # Save guide aliases to data/processed/ (cls_*.tif)
    print("Saving guide compatibility rasters to data/processed/ (cls_*.tif)...")
    save_tif(PROC / "cls_slope.tif", binned_layers["slope"], T, nodata=0)
    save_tif(PROC / "cls_curvature.tif", binned_layers["curvature"], T, nodata=0)
    save_tif(PROC / "cls_curv.tif", binned_layers["curvature"], T, nodata=0)
    save_tif(PROC / "cls_dist_roads.tif", binned_layers["dist_roads"], T, nodata=0)
    save_tif(PROC / "cls_d_road.tif", binned_layers["dist_roads"], T, nodata=0)
    save_tif(PROC / "cls_dist_streams.tif", binned_layers["dist_streams"], T, nodata=0)
    save_tif(PROC / "cls_d_stream.tif", binned_layers["dist_streams"], T, nodata=0)
    save_tif(PROC / "cls_aspect.tif", binned_layers["aspect"], T, nodata=0)
    save_tif(PROC / "cls_landcover.tif", binned_layers["landcover"], T, nodata=0)
    save_tif(PROC / "cls_landcover_contiguous.tif", binned_layers["landcover_contiguous"], T, nodata=0)
    save_tif(PROC / "cls_landcover_esa.tif", binned_layers["landcover_esa"], T, nodata=0)

    # Build and export bin edges summary table
    bin_edge_rows = []
    for factor_name, b_info in BIN_DEFINITIONS.items():
        edges = b_info["edges"]
        labels = b_info["labels"]
        for class_id, label in labels.items():
            bin_edge_rows.append({
                "factor": factor_name,
                "class_id": class_id,
                "class_label": label,
                "description": b_info["description"],
                "bin_edges": str(edges) if edges is not None else "precomputed",
            })
    for c_id, name in CONTIGUOUS_NAMES.items():
        bin_edge_rows.append({
            "factor": "landcover",
            "class_id": c_id,
            "class_label": f"{name} (ESA code {CONTIGUOUS_TO_ESA[c_id]})",
            "description": "ESA WorldCover 2021 categorical",
            "bin_edges": f"code={CONTIGUOUS_TO_ESA[c_id]}",
        })

    bin_df = pd.DataFrame(bin_edge_rows)
    bin_df.to_csv(OUT / "bin_edges.csv", index=False)
    bin_df.to_csv(TABLES_DIR / "bin_edges.csv", index=False)
    print(f"Exported bin definitions to {OUT / 'bin_edges.csv'} and {TABLES_DIR / 'bin_edges.csv'}")

    # Verification and summary report
    print("\n" + "=" * 78)
    print("BINNED FACTOR DISTRIBUTION VERIFICATION (INSIDE DISTRICT)")
    print("=" * 78)
    for name in ["slope", "curvature", "dist_roads", "dist_streams", "aspect", "landcover", "landcover_esa"]:
        arr = binned_layers[name]
        dist_vals = arr[mask]
        u_vals, counts = np.unique(dist_vals, return_counts=True)
        outside_zero = bool((arr[~mask] == 0).all())
        unassigned_in_dist = int((dist_vals == 0).sum())

        print(f"\nFactor: {name} (dtype={arr.dtype}, nodata=0, outside_cells_all_zero={outside_zero})")
        print(f"  Unassigned cells in district: {unassigned_in_dist}")
        for v, cnt in zip(u_vals, counts):
            pct = cnt / n_dist * 100.0
            print(f"    Class {v:2d}: {cnt:9,d} cells ({pct:5.2f}%)")

        assert unassigned_in_dist == 0, f"Found unassigned cells in district for {name}"
        assert outside_zero, f"Non-zero cells outside district for {name}"

    # Specific bin assertions
    assert set(np.unique(binned_layers["slope"][mask])) == {1, 2, 3, 4, 5}
    assert set(np.unique(binned_layers["curvature"][mask])) == {1, 2, 3}
    assert set(np.unique(binned_layers["dist_roads"][mask])) == {1, 2, 3, 4, 5}
    assert set(np.unique(binned_layers["dist_streams"][mask])) == {1, 2, 3, 4, 5}
    assert set(np.unique(binned_layers["aspect"][mask])) == set(range(1, 10))
    assert set(np.unique(binned_layers["landcover"][mask])).issubset(set(range(1, 9)))
    assert set(np.unique(binned_layers["landcover_contiguous"][mask])).issubset(set(range(1, 9)))
    assert set(np.unique(binned_layers["landcover_esa"][mask])).issubset(set(ESA_LEGEND.keys()))

    print("\n" + "=" * 78)
    print("ALL CONDITIONING FACTOR BINNING CHECKS PASSED SUCCESSFULLY!")
    print("=" * 78 + "\n")
    return binned_layers


if __name__ == "__main__":
    bin_conditioning_factors()
