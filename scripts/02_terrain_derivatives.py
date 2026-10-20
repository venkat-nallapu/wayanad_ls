"""02_terrain_derivatives.py - Terrain Derivative Extraction (Slope, Aspect, Curvature).

Computes morphometric factors from the projected 30m Copernicus DEM:
1. Horn's Slope (in degrees) via 3x3 convolution
2. Aspect (downslope compass bearing 0-360 deg, -1 for flat slopes < 1 deg)
3. Total Curvature (2D Laplacian operator)
4. Orientation verification via synthetic tilted plane tests (N, NE, E, SE, S, SW, W, NW)
5. Saves continuous factor GeoTIFFs to data/processed/factors_continuous/
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import convolve

from common import (
    PROC, MASTER_DIR, FACTORS_CONT_DIR, FACTORS_BIN_DIR, RES,
    load_grid, load_tif, save_tif
)

# Horn's 3x3 convolution kernels for metric resolution
# When convolved with scipy.ndimage.convolve (which inverts kernels 180 degrees):
# KX computes dz/dx (elevation rise toward East: positive)
# KY computes dz/dy_img (elevation rise toward South down rows: positive)
KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / (8.0 * RES)
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / (8.0 * RES)
K_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / (RES ** 2)

# Aspect compass bin mapping definitions
ASPECT_COMPASS_NAMES = {
    1: "North (N)",
    2: "Northeast (NE)",
    3: "East (E)",
    4: "Southeast (SE)",
    5: "South (S)",
    6: "Southwest (SW)",
    7: "West (W)",
    8: "Northwest (NW)",
    9: "Flat",
}


def compute_aspect_bins(aspect, slope):
    """Bin continuous aspect into 9 discrete compass classes:
    1: North (337.5° - 22.5°)
    2: Northeast (22.5° - 67.5°)
    3: East (67.5° - 112.5°)
    4: Southeast (112.5° - 157.5°)
    5: South (157.5° - 202.5°)
    6: Southwest (202.5° - 247.5°)
    7: West (247.5° - 292.5°)
    8: Northwest (292.5° - 337.5°)
    9: Flat (slope < 1.0° or aspect < 0)
    0: NoData / unclassified
    """
    aspect = np.asarray(aspect, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)
    valid_mask = np.isfinite(aspect) & np.isfinite(slope)
    
    binned = np.zeros(aspect.shape, dtype=np.uint8)
    
    # 8 compass directions: ((deg + 22.5) % 360) // 45 gives 0..7, so +1 gives 1..8
    safe_asp = np.where(valid_mask & (aspect >= 0), aspect, 0.0)
    dirs = (((safe_asp + 22.5) % 360.0) // 45.0).astype(np.uint8) + 1
    binned[valid_mask] = dirs[valid_mask]
    
    # Flat terrain assignment
    flat_mask = valid_mask & ((slope < 1.0) | (aspect < 0))
    binned[flat_mask] = 9
    
    return binned


def run_synthetic_plane_test():
    """Verify aspect orientation correctness and compass binning against synthetic sloping planes."""
    print("Running synthetic plane aspect orientation and compass binning tests...")
    test_res = RES
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / (8.0 * test_res)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / (8.0 * test_res)

    # Grid of coordinates: r increases South (down rows), c increases East (right columns)
    r = np.arange(50)[:, None]
    c = np.arange(50)[None, :]

    directions = [
        ("North", 0.0, 1),
        ("Northeast", 45.0, 2),
        ("East", 90.0, 3),
        ("Southeast", 135.0, 4),
        ("South", 180.0, 5),
        ("Southwest", 225.0, 6),
        ("West", 270.0, 7),
        ("Northwest", 315.0, 8),
    ]

    for name, expected_azimuth, expected_bin in directions:
        rad = np.radians(expected_azimuth)
        # Plane drops in direction expected_azimuth (downslope compass heading)
        # Downslope vector in (East, North) is (sin(rad), cos(rad))
        # Since r increases South (North is -r), coordinate is (c * res, -r * res)
        plane = 1000.0 - (c * test_res * np.sin(rad) - r * test_res * np.cos(rad)) * 0.15

        dx = convolve(plane.astype(np.float32), kx, mode="nearest")
        dy = convolve(plane.astype(np.float32), ky, mode="nearest")

        computed_aspect = float((np.degrees(np.arctan2(-dx, dy)) % 360)[25, 25])
        slope_deg = float(np.degrees(np.arctan(np.hypot(dx, dy)))[25, 25])
        computed_bin = int(compute_aspect_bins(np.array([[computed_aspect]]), np.array([[slope_deg]]))[0, 0])

        diff = abs(computed_aspect - expected_azimuth)
        if diff > 180:
            diff = 360 - diff

        assert diff < 1e-3, (
            f"Orientation test FAILED for {name}: expected {expected_azimuth} deg, "
            f"got {computed_aspect:.2f} deg"
        )
        assert abs(slope_deg - np.degrees(np.arctan(0.15))) < 1e-3, (
            f"Slope test FAILED for {name}: expected {np.degrees(np.arctan(0.15)):.2f} deg, "
            f"got {slope_deg:.2f} deg"
        )
        assert computed_bin == expected_bin, (
            f"Compass bin test FAILED for {name}: expected bin {expected_bin}, got {computed_bin}"
        )
        print(f"  [PASS] {name:<10s} -> Azimuth: {computed_aspect:5.1f} deg | Slope: {slope_deg:4.2f} deg | Bin: {computed_bin} ({ASPECT_COMPASS_NAMES[computed_bin]})")

    # Test flat plane
    flat_plane = np.full((50, 50), 500.0, dtype=np.float32)
    dx = convolve(flat_plane, kx, mode="nearest")
    dy = convolve(flat_plane, ky, mode="nearest")
    flat_slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    flat_aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)
    flat_aspect[flat_slope < 1.0] = -1.0
    flat_bin = int(compute_aspect_bins(flat_aspect, flat_slope)[25, 25])
    assert flat_bin == 9, f"Flat plane bin expected 9, got {flat_bin}"
    print(f"  [PASS] Flat Plane -> Slope: {flat_slope[25, 25]:.2f} deg | Aspect: {flat_aspect[25, 25]:.1f} | Bin: {flat_bin} (Flat)")

    print("All synthetic plane orientation and binning tests PASSED!\n")


def compute_terrain_derivatives():
    # 1. Run orientation unit test before processing
    run_synthetic_plane_test()

    # 2. Load master grid and DEM
    T, H, W = load_grid()
    dem_path = MASTER_DIR / "dem_30m.tif"
    mask_path = MASTER_DIR / "mask.tif"

    print(f"Loading DEM from {dem_path}...")
    dem = load_tif(dem_path).astype(np.float32)
    mask = load_tif(mask_path).astype(bool)

    assert dem.shape == (H, W), f"DEM shape {dem.shape} does not match master grid ({H}, {W})"
    assert mask.shape == (H, W), f"Mask shape {mask.shape} does not match master grid ({H}, {W})"

    # 3. Compute elevation gradients (dz/dx, dz/dy) via scipy.ndimage.convolve
    print("Computing elevation gradients (dz/dx, dz/dy) via scipy.ndimage.convolve...")
    dzdx = convolve(dem, KX, mode="nearest")
    dzdy = convolve(dem, KY, mode="nearest")

    # 4. Slope calculation (Horn's method, in degrees)
    print("Computing Horn's slope in degrees...")
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)

    # 5. Aspect calculation (compass azimuth 0-360 deg clockwise from North)
    print("Computing aspect azimuth in degrees...")
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360).astype(np.float32)
    # Set flat terrain (slope < 1.0 deg) to -1.0 (no defined aspect)
    aspect[slope < 1.0] = -1.0

    # 6. Compass binning (1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 9=flat)
    print("Computing aspect compass bins (1-9)...")
    aspect_binned = compute_aspect_bins(aspect, slope)

    # 7. Curvature calculation (2D Laplacian)
    # Positive = concave / valley, Negative = convex / ridge
    print("Computing profile/plan total curvature (Laplacian)...")
    curv = convolve(dem, K_LAP, mode="nearest").astype(np.float32)

    # 8. Save continuous factor GeoTIFFs to data/processed/factors_continuous/
    print("\nSaving continuous factor GeoTIFFs...")
    save_tif(FACTORS_CONT_DIR / "slope.tif", slope, T, nodata=np.nan)
    save_tif(FACTORS_CONT_DIR / "aspect.tif", aspect, T, nodata=np.nan)
    save_tif(FACTORS_CONT_DIR / "curv.tif", curv, T, nodata=np.nan)
    save_tif(FACTORS_CONT_DIR / "curvature.tif", curv, T, nodata=np.nan)

    # Save to base processed directory for guide compatibility
    save_tif(PROC / "slope.tif", slope, T, nodata=np.nan)
    save_tif(PROC / "aspect.tif", aspect, T, nodata=np.nan)
    save_tif(PROC / "curv.tif", curv, T, nodata=np.nan)

    # 9. Save binned aspect GeoTIFFs to data/processed/factors_binned/
    print("Saving binned aspect GeoTIFFs...")
    save_tif(FACTORS_BIN_DIR / "aspect.tif", aspect_binned, T, nodata=0)
    save_tif(FACTORS_BIN_DIR / "aspect_binned.tif", aspect_binned, T, nodata=0)
    save_tif(PROC / "cls_aspect.tif", aspect_binned, T, nodata=0)

    # 10. Compute and verify statistics within district domain
    slope_dist = slope[mask]
    aspect_dist = aspect[mask]
    aspect_bin_dist = aspect_binned[mask]
    curv_dist = curv[mask]

    flat_count = int((aspect_dist == -1.0).sum())
    valid_count = int(mask.sum())

    print("\n" + "=" * 70)
    print("TERRAIN DERIVATIVES STATISTICAL VERIFICATION")
    print("=" * 70)
    print(f"{'Factor':<18} {'Domain':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    print(f"{'Slope (deg)':<18} {'District':<10} {np.min(slope_dist):<10.2f} {np.max(slope_dist):<10.2f} {np.mean(slope_dist):<10.2f} {np.std(slope_dist):<10.2f}")
    print(f"{'Slope (deg)':<18} {'Full Grid':<10} {np.min(slope):<10.2f} {np.max(slope):<10.2f} {np.mean(slope):<10.2f} {np.std(slope):<10.2f}")
    print(f"{'Aspect (deg)':<18} {'District':<10} {np.min(aspect_dist):<10.2f} {np.max(aspect_dist):<10.2f} {np.mean(aspect_dist[aspect_dist >= 0]):<10.2f} {np.std(aspect_dist[aspect_dist >= 0]):<10.2f}")
    print(f"{'Curvature':<18} {'District':<10} {np.min(curv_dist):<10.4f} {np.max(curv_dist):<10.4f} {np.mean(curv_dist):<10.4e} {np.std(curv_dist):<10.4f}")
    print(f"{'Curvature':<18} {'Full Grid':<10} {np.min(curv):<10.4f} {np.max(curv):<10.4f} {np.mean(curv):<10.4e} {np.std(curv):<10.4f}")
    print("-" * 70)
    print(f"Flat Terrain Pixels (Slope < 1 deg): {flat_count:,} / {valid_count:,} ({flat_count / valid_count * 100:.2f}%)")
    print("-" * 70)
    print("Aspect Compass Bin Distribution inside District:")
    for b_idx in range(1, 10):
        cnt = int((aspect_bin_dist == b_idx).sum())
        pct = cnt / valid_count * 100
        print(f"  Bin {b_idx} ({ASPECT_COMPASS_NAMES[b_idx]:<16s}): {cnt:8,d} cells ({pct:5.2f}%)")
    print("=" * 70)

    # Verification assertions
    assert 0.0 <= np.min(slope_dist) < 1.0, f"Unexpected minimum slope: {np.min(slope_dist)}"
    assert 45.0 < np.max(slope_dist) < 80.0, f"Unexpected maximum slope: {np.max(slope_dist)}"
    assert np.min(aspect_dist) == -1.0, f"Expected -1.0 for flat aspect, got: {np.min(aspect_dist)}"
    assert 355.0 < np.max(aspect_dist) <= 360.0, f"Aspect max out of bounds: {np.max(aspect_dist)}"
    assert np.min(curv_dist) < -0.01, f"Curvature min expected negative, got: {np.min(curv_dist)}"
    assert np.max(curv_dist) > 0.01, f"Curvature max expected positive, got: {np.max(curv_dist)}"
    assert abs(np.mean(curv_dist)) < 1e-3, f"Curvature mean expected ~0, got: {np.mean(curv_dist)}"
    assert np.isnan(slope_dist).sum() == 0, "NaNs detected in district slope!"
    assert np.isnan(aspect_dist).sum() == 0, "NaNs detected in district aspect!"
    assert np.isnan(curv_dist).sum() == 0, "NaNs detected in district curvature!"
    assert set(np.unique(aspect_bin_dist)) == set(range(1, 10)), "Not all 9 aspect bins populated in district!"
    print("ALL DERIVATIVE VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    compute_terrain_derivatives()
