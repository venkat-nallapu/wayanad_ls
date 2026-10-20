"""06_dataset_split.py - Landslide Dataset Split and Non-Landslide Sampling (Step 9).

1. Loads 204 Wayanad landslide points within the district mask.
2. Partitions into 70% Training (143 points) and 30% Validation (61 points) using fixed SEED = 42.
3. Generates 1:1 non-landslide points (143 train negative, 61 val negative) strictly inside
   mask.tif and at least 300 meters away from ANY of the 204 known landslide points.
4. Converts coordinates to grid row/col indices.
5. Saves sample sets to:
   - data/processed/samples/ (GeoJSON, CSV, NPZ)
   - data/processed/points.npz (Guide Step 10 compatibility)
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy, rowcol
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from common import (
    RAW, PROC, MASTER_DIR, SAMPLES_DIR, CRS, RES, SEED, TRAIN_FRAC, NONSLIDE_MIN_DIST,
    load_grid, load_tif
)


def split_and_sample_dataset(seed=SEED, train_frac=TRAIN_FRAC, min_dist=NONSLIDE_MIN_DIST):
    print("=" * 78)
    print("STEP 9: LANDSLIDE DATASET SPLIT & NON-LANDSLIDE SAMPLING")
    print("=" * 78)

    T, H, W = load_grid()
    mask_path = MASTER_DIR / "mask.tif"
    mask = load_tif(mask_path).astype(bool)
    n_dist = int(mask.sum())

    # 1. Load landslide inventory
    inv_path = RAW / "inventory" / "wayanad_landslides.geojson"
    if not inv_path.exists():
        raise FileNotFoundError(f"Landslide inventory not found at {inv_path}")

    inv_gdf = gpd.read_file(inv_path)
    if inv_gdf.crs != CRS:
        inv_gdf = inv_gdf.to_crs(CRS)

    print(f"Loaded {len(inv_gdf)} raw landslide features. CRS: {inv_gdf.crs}")

    # Sample mask to filter points strictly within district mask
    coords_all = [(geom.x, geom.y) for geom in inv_gdf.geometry]
    with rasterio.open(mask_path) as src:
        m_vals = np.array([val[0] for val in src.sample(coords_all)])

    inv_dist = inv_gdf[m_vals == 1].copy().reset_index(drop=True)
    n_total = len(inv_dist)
    print(f"Landslides strictly within district mask: {n_total} points.")
    assert n_total == 204, f"Expected exactly 204 landslides inside district mask, found {n_total}"

    # Extract coordinates and (row, col) indices
    pos_x = inv_dist.geometry.x.values
    pos_y = inv_dist.geometry.y.values
    pos_xy = np.column_stack([pos_x, pos_y])

    pos_rows, pos_cols = rowcol(T, pos_x, pos_y)
    pos_rc = np.column_stack([np.array(pos_rows), np.array(pos_cols)])

    # 2. Partition 70% Train (143) and 30% Validation (61) using fixed SEED
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_tr = int(round(train_frac * n_total))
    n_val = n_total - n_tr
    print(f"Partition: {n_tr} Training landslides (70%), {n_val} Validation landslides (30%). Seed: {seed}")
    assert n_tr == 143, f"Expected 143 train landslides, got {n_tr}"
    assert n_val == 61, f"Expected 61 val landslides, got {n_val}"

    train_pos_idx = perm[:n_tr]
    val_pos_idx = perm[n_tr:]

    train_pos_rc = pos_rc[train_pos_idx]
    train_pos_xy = pos_xy[train_pos_idx]

    val_pos_rc = pos_rc[val_pos_idx]
    val_pos_xy = pos_xy[val_pos_idx]

    # 3. Non-landslide sampling: strictly inside mask and >= 300m from ANY of the 204 landslides
    print(f"\nComputing Euclidean distance transform from all {n_total} landslides...")
    slide_cells = np.zeros((H, W), dtype=bool)
    slide_cells[pos_rc[:, 0], pos_rc[:, 1]] = True

    # EDT in meters
    dist_to_slides = distance_transform_edt(~slide_cells) * RES

    # Buffer by cell diagonal (sqrt(2) * 30m ~= 42.4m) so cell center is >= 350m
    # ensuring worst-case point-to-point distance strictly exceeds min_dist (300m)
    cand_mask = (dist_to_slides >= (min_dist + 50.0)) & mask
    cand_rc = np.argwhere(cand_mask)
    n_cand = len(cand_rc)
    print(f"Candidate non-landslide pixels (>={min_dist + 50}m from any landslide): {n_cand:,} / {n_dist:,} cells")
    assert n_cand >= n_total, f"Not enough candidate non-landslide cells: {n_cand} < {n_total}"

    # Sample 204 non-landslide cells without replacement
    neg_choice_idx = rng.choice(n_cand, size=n_total, replace=False)
    neg_rc = cand_rc[neg_choice_idx]

    # Get coordinate centers for negative samples
    neg_xs, neg_ys = xy(T, neg_rc[:, 0], neg_rc[:, 1])
    neg_xy = np.column_stack([neg_xs, neg_ys])

    # Rigorous verification: compute exact pairwise Euclidean distance from each neg to all 204 pos
    pairwise_dists = cdist(neg_xy, pos_xy)
    min_dist_observed = float(np.min(pairwise_dists))
    print(f"Minimum observed point-to-point distance from any non-landslide to any landslide: {min_dist_observed:.2f} m")
    assert min_dist_observed >= min_dist, (
        f"Exclusion distance violated! Min observed: {min_dist_observed:.2f} m < {min_dist} m"
    )

    # Partition negatives 1:1 with positives
    train_neg_rc = neg_rc[:n_tr]
    train_neg_xy = neg_xy[:n_tr]

    val_neg_rc = neg_rc[n_tr:]
    val_neg_xy = neg_xy[n_tr:]

    # 4. Save points to NPZ
    npz_data = {
        "train": train_pos_rc,          # shape (143, 2) [row, col]
        "val": val_pos_rc,              # shape (61, 2)  [row, col]
        "neg_tr": train_neg_rc,        # shape (143, 2) [row, col]
        "neg_val": val_neg_rc,          # shape (61, 2)  [row, col]
        "train_xy": train_pos_xy,       # shape (143, 2) [x, y]
        "val_xy": val_pos_xy,           # shape (61, 2)  [x, y]
        "neg_tr_xy": train_neg_xy,      # shape (143, 2) [x, y]
        "neg_val_xy": val_neg_xy,       # shape (61, 2)  [x, y]
    }

    samples_npz_path = SAMPLES_DIR / "points.npz"
    np.savez(samples_npz_path, **npz_data)
    # Also save to data/processed/points.npz for direct guide Step 10 compatibility
    np.savez(PROC / "points.npz", **npz_data)
    print(f"\nSaved sample matrices to {samples_npz_path} and {PROC / 'points.npz'}")

    # 5. Build comprehensive GeoDataFrames & CSVs
    records = []
    
    # Positive train
    for i, ((r, c), (x_coord, y_coord)) in enumerate(zip(train_pos_rc, train_pos_xy)):
        records.append({
            "sample_id": f"TR_POS_{i+1:03d}",
            "split": "train",
            "label": 1,
            "class_name": "landslide",
            "row": int(r),
            "col": int(c),
            "x_utm": float(x_coord),
            "y_utm": float(y_coord),
            "geometry": Point(x_coord, y_coord),
        })

    # Negative train
    for i, ((r, c), (x_coord, y_coord)) in enumerate(zip(train_neg_rc, train_neg_xy)):
        records.append({
            "sample_id": f"TR_NEG_{i+1:03d}",
            "split": "train",
            "label": 0,
            "class_name": "non_landslide",
            "row": int(r),
            "col": int(c),
            "x_utm": float(x_coord),
            "y_utm": float(y_coord),
            "geometry": Point(x_coord, y_coord),
        })

    # Positive val
    for i, ((r, c), (x_coord, y_coord)) in enumerate(zip(val_pos_rc, val_pos_xy)):
        records.append({
            "sample_id": f"VAL_POS_{i+1:03d}",
            "split": "val",
            "label": 1,
            "class_name": "landslide",
            "row": int(r),
            "col": int(c),
            "x_utm": float(x_coord),
            "y_utm": float(y_coord),
            "geometry": Point(x_coord, y_coord),
        })

    # Negative val
    for i, ((r, c), (x_coord, y_coord)) in enumerate(zip(val_neg_rc, val_neg_xy)):
        records.append({
            "sample_id": f"VAL_NEG_{i+1:03d}",
            "split": "val",
            "label": 0,
            "class_name": "non_landslide",
            "row": int(r),
            "col": int(c),
            "x_utm": float(x_coord),
            "y_utm": float(y_coord),
            "geometry": Point(x_coord, y_coord),
        })

    all_gdf = gpd.GeoDataFrame(records, crs=CRS)

    # Save all samples
    all_gdf.to_file(SAMPLES_DIR / "samples_all.geojson", driver="GeoJSON")
    all_gdf.drop(columns="geometry").to_csv(SAMPLES_DIR / "samples_all.csv", index=False)

    # Save split-specific files
    train_gdf = all_gdf[all_gdf.split == "train"].copy()
    val_gdf = all_gdf[all_gdf.split == "val"].copy()

    train_gdf.to_file(SAMPLES_DIR / "train_samples.geojson", driver="GeoJSON")
    train_gdf.drop(columns="geometry").to_csv(SAMPLES_DIR / "train_samples.csv", index=False)

    val_gdf.to_file(SAMPLES_DIR / "val_samples.geojson", driver="GeoJSON")
    val_gdf.drop(columns="geometry").to_csv(SAMPLES_DIR / "val_samples.csv", index=False)

    # Individual subsets
    all_gdf[(all_gdf.split == "train") & (all_gdf.label == 1)].to_file(
        SAMPLES_DIR / "train_landslides.geojson", driver="GeoJSON"
    )
    all_gdf[(all_gdf.split == "train") & (all_gdf.label == 0)].to_file(
        SAMPLES_DIR / "train_negatives.geojson", driver="GeoJSON"
    )
    all_gdf[(all_gdf.split == "val") & (all_gdf.label == 1)].to_file(
        SAMPLES_DIR / "val_landslides.geojson", driver="GeoJSON"
    )
    all_gdf[(all_gdf.split == "val") & (all_gdf.label == 0)].to_file(
        SAMPLES_DIR / "val_negatives.geojson", driver="GeoJSON"
    )

    print(f"Exported GeoJSON and CSV sample sets to {SAMPLES_DIR}/")

    # Print summary table
    print("\n" + "=" * 78)
    print("DATASET SPLIT & SAMPLE SUMMARY")
    print("=" * 78)
    print(f"{'Dataset Split':<18} {'Landslides (1)':<18} {'Non-Landslides (0)':<20} {'Total':<10}")
    print("-" * 78)
    print(f"{'Training (70%)':<18} {len(train_pos_rc):<18d} {len(train_neg_rc):<20d} {len(train_pos_rc)+len(train_neg_rc):<10d}")
    print(f"{'Validation (30%)':<18} {len(val_pos_rc):<18d} {len(val_neg_rc):<20d} {len(val_pos_rc)+len(val_neg_rc):<10d}")
    print("-" * 78)
    print(f"{'Total (100%)':<18} {n_total:<18d} {n_total:<20d} {2*n_total:<10d}")
    print("=" * 78)
    print("STEP 9 DATASET SPLIT COMPLETED SUCCESSFULLY!\n")

    return npz_data, all_gdf


if __name__ == "__main__":
    split_and_sample_dataset()
