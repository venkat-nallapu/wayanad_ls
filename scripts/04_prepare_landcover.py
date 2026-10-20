"""04_prepare_landcover.py - ESA WorldCover 10m Reprojection and Alignment.

1. Loads master metric grid definition (UTM Zone 43N, EPSG:32643, 30m resolution).
2. Warps 10m ESA WorldCover 2021 raster onto the 30m master grid using Resampling.mode
   (mode resampling preserves discrete categorical land cover classes).
3. Masks pixels outside the district boundary to 0 (nodata).
4. Computes land cover class distributions and verifies complete district coverage.
5. Outputs:
   - data/processed/factors_binned/landcover.tif
   - data/processed/factors_continuous/landcover.tif
   - data/processed/landcover.tif
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rasterio.warp import Resampling

from common import (
    RAW, PROC, MASTER_DIR, FACTORS_BIN_DIR, FACTORS_CONT_DIR,
    load_grid, load_tif, save_tif, to_grid
)

# ESA WorldCover 2021 official legend definitions
WORLDCOVER_LEGEND = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def prepare_landcover():
    T, H, W = load_grid()
    mask_path = MASTER_DIR / "mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"District mask not found at {mask_path}")
    mask = load_tif(mask_path).astype(bool)

    lc_tiles = sorted((RAW / "landcover").glob("*.tif"))
    if not lc_tiles:
        raise FileNotFoundError(f"No ESA WorldCover GeoTIFFs found in {RAW / 'landcover'}")

    print(f"Found {len(lc_tiles)} land cover tile(s) in {RAW / 'landcover'}:")
    for t in lc_tiles:
        print(f"  - {t.name}")

    # Warp and mosaic onto master grid using Resampling.mode (categorical)
    lc = np.zeros((H, W), dtype="uint8")
    for tile_path in lc_tiles:
        print(f"Resampling {tile_path.name} to 30m master grid via Resampling.mode...")
        tile_warped = to_grid(
            tile_path,
            T,
            H,
            W,
            resampling=Resampling.mode,
            dtype="uint8",
            nodata=0,
        )
        lc = np.where(lc == 0, tile_warped, lc).astype("uint8")

    # Mask outside pixels to 0
    lc_masked = np.where(mask, lc, 0).astype("uint8")

    print("\nSaving land cover factor GeoTIFFs...")
    save_tif(FACTORS_BIN_DIR / "landcover.tif", lc_masked, T, nodata=0)
    save_tif(FACTORS_CONT_DIR / "landcover.tif", lc_masked, T, nodata=0)
    save_tif(PROC / "landcover.tif", lc_masked, T, nodata=0)

    # Statistical verification inside district boundary
    lc_district = lc_masked[mask]
    total_valid = len(lc_district)
    unassigned = int((lc_district == 0).sum())
    unique_classes, counts = np.unique(lc_district, return_counts=True)

    print("\n" + "=" * 70)
    print("ESA WORLDCOVER (30m) DISTRICT COMPOSITION VERIFICATION")
    print("=" * 70)
    print(f"{'Class Code':<12} {'Description':<28} {'Cell Count':<14} {'District %':<10}")
    print("-" * 70)
    for code, cnt in zip(unique_classes, counts):
        desc = WORLDCOVER_LEGEND.get(int(code), "Unknown / Unclassified")
        pct = (cnt / total_valid) * 100.0
        print(f"{code:<12} {desc:<28} {cnt:<14,d} {pct:<10.2f}%")
    print("-" * 70)
    print(f"Total District Cells:   {total_valid:,}")
    print(f"Unassigned Cells (=0):  {unassigned:,}")
    print(f"Outside Cells All Zero: {bool((lc_masked[~mask] == 0).all())}")
    print("=" * 70)

    # Verification assertions
    assert lc_masked.shape == (H, W), f"Land cover shape {lc_masked.shape} != {(H, W)}"
    assert lc_masked.dtype == np.uint8, f"Unexpected dtype: {lc_masked.dtype}"
    assert unassigned == 0, f"Found {unassigned} unassigned cells within district boundary!"
    assert (lc_masked[~mask] == 0).all(), "Non-zero cells found outside district boundary!"
    assert set(unique_classes).issubset(set(WORLDCOVER_LEGEND.keys())), "Found invalid class codes!"
    
    # In Wayanad (Western Ghats), Tree cover (10) should dominate (>70%)
    tree_pct = (int((lc_district == 10).sum()) / total_valid) * 100.0
    assert tree_pct > 70.0, f"Expected tree cover >70%, found {tree_pct:.2f}%"

    print("ALL LAND COVER VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
    return lc_masked


if __name__ == "__main__":
    prepare_landcover()
