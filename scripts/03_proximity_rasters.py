"""03_proximity_rasters.py - Euclidean Distance Transforms to Roads and Streams.

Computes continuous distance (in meters) to nearest road and stream:
1. Loads master grid definition (UTM Zone 43N, EPSG:32643, 30m resolution).
2. Reprojects OSM line vectors (roads and streams) to EPSG:32643.
3. Rasterizes line geometries onto a padded master grid (PAD = 100 cells / 3000m)
   using all_touched=True to avoid border bias.
4. Computes Euclidean distance transforms (scipy.ndimage.distance_transform_edt * RES).
5. Crops padded raster back to (H, W) master grid and masks outside pixels to NaN.
6. Outputs:
   - data/processed/factors_continuous/dist_roads.tif
   - data/processed/factors_continuous/dist_streams.tif
   - data/processed/factors_continuous/d_road.tif (guide alias)
   - data/processed/factors_continuous/d_stream.tif (guide alias)
   - data/processed/dist_roads.tif, dist_streams.tif, d_road.tif, d_stream.tif
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

from common import (
    RAW, PROC, MASTER_DIR, FACTORS_CONT_DIR, CRS, RES,
    load_grid, load_tif, save_tif
)

PAD = 100  # Padded border pixels (3000m at 30m resolution)


def compute_proximity(gdf, T, H, W, pad=PAD, res=RES):
    """Compute Euclidean distance transform raster in meters from line vector geometries."""
    # Ensure metric projected CRS
    if gdf.crs != CRS:
        gdf = gdf.to_crs(CRS)

    # Filter non-empty linear geometries (LineString, MultiLineString)
    valid_gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geom_type.isin(["LineString", "MultiLineString"])
    ]
    if len(valid_gdf) == 0:
        raise ValueError("No valid LineString or MultiLineString geometries found in input GeoDataFrame.")

    # Define padded affine transform and dimensions
    Tp = Affine(T.a, 0.0, T.c - pad * res, 0.0, T.e, T.f + pad * res)
    Hp, Wp = H + 2 * pad, W + 2 * pad

    # Rasterize lines with all_touched=True
    shapes = ((geom, 1) for geom in valid_gdf.geometry)
    burned = rasterize(
        shapes,
        out_shape=(Hp, Wp),
        transform=Tp,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    burned_count = int(burned.sum())
    if burned_count == 0:
        raise ValueError("No linear features were rasterized onto the grid (burned_count == 0).")

    # Distance transform: 0 on lines, increasing with Euclidean distance
    # distance_transform_edt computes distance from cells where condition (burned == 0) is True
    # to the nearest cell where condition is False (i.e. burned == 1).
    edt = distance_transform_edt(burned == 0) * res

    # Crop padded margin back to master grid dimensions
    cropped = edt[pad:-pad, pad:-pad].astype(np.float32)

    return cropped, burned_count


def generate_proximity_rasters():
    T, H, W = load_grid()
    mask_path = MASTER_DIR / "mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"District mask not found at {mask_path}")
    mask = load_tif(mask_path).astype(bool)

    roads_path = RAW / "osm" / "wayanad_roads.geojson"
    streams_path = RAW / "osm" / "wayanad_streams.geojson"

    if not roads_path.exists():
        raise FileNotFoundError(f"Roads vector data not found at {roads_path}")
    if not streams_path.exists():
        raise FileNotFoundError(f"Streams vector data not found at {streams_path}")

    # --- 1. Roads Proximity ---
    print(f"Loading roads from {roads_path}...")
    roads_gdf = gpd.read_file(roads_path)
    print(f"Loaded {len(roads_gdf)} road features. Computing distance transform (PAD={PAD})...")
    dist_roads_full, burned_roads = compute_proximity(roads_gdf, T, H, W, pad=PAD, res=RES)
    print(f"Burned road cells in padded grid: {burned_roads:,}")

    # Mask pixels outside the district boundary to NaN
    dist_roads_masked = np.where(mask, dist_roads_full, np.nan).astype(np.float32)

    print("Saving road proximity rasters...")
    save_tif(FACTORS_CONT_DIR / "dist_roads.tif", dist_roads_masked, T, nodata=np.nan)
    save_tif(FACTORS_CONT_DIR / "d_road.tif", dist_roads_masked, T, nodata=np.nan)
    save_tif(PROC / "dist_roads.tif", dist_roads_masked, T, nodata=np.nan)
    save_tif(PROC / "d_road.tif", dist_roads_masked, T, nodata=np.nan)

    # --- 2. Streams Proximity ---
    print(f"\nLoading streams from {streams_path}...")
    streams_gdf = gpd.read_file(streams_path)
    print(f"Loaded {len(streams_gdf)} stream features. Computing distance transform (PAD={PAD})...")
    dist_streams_full, burned_streams = compute_proximity(streams_gdf, T, H, W, pad=PAD, res=RES)
    print(f"Burned stream cells in padded grid: {burned_streams:,}")

    # Mask pixels outside the district boundary to NaN
    dist_streams_masked = np.where(mask, dist_streams_full, np.nan).astype(np.float32)

    print("Saving stream proximity rasters...")
    save_tif(FACTORS_CONT_DIR / "dist_streams.tif", dist_streams_masked, T, nodata=np.nan)
    save_tif(FACTORS_CONT_DIR / "d_stream.tif", dist_streams_masked, T, nodata=np.nan)
    save_tif(PROC / "dist_streams.tif", dist_streams_masked, T, nodata=np.nan)
    save_tif(PROC / "d_stream.tif", dist_streams_masked, T, nodata=np.nan)

    # --- 3. Statistical Verification and Assertions ---
    dist_roads_valid = dist_roads_masked[mask]
    dist_streams_valid = dist_streams_masked[mask]

    r_min, r_max = float(np.nanmin(dist_roads_valid)), float(np.nanmax(dist_roads_valid))
    r_mean, r_median = float(np.nanmean(dist_roads_valid)), float(np.median(dist_roads_valid))
    r_std = float(np.nanstd(dist_roads_valid))

    s_min, s_max = float(np.nanmin(dist_streams_valid)), float(np.nanmax(dist_streams_valid))
    s_mean, s_median = float(np.nanmean(dist_streams_valid)), float(np.median(dist_streams_valid))
    s_std = float(np.nanstd(dist_streams_valid))

    print("\n" + "=" * 75)
    print("PROXIMITY RASTERS STATISTICAL VERIFICATION")
    print("=" * 75)
    print(f"{'Factor':<18} {'Min (m)':<10} {'Max (m)':<10} {'Mean (m)':<10} {'Median (m)':<12} {'Std (m)':<10}")
    print("-" * 75)
    print(f"{'Dist Roads':<18} {r_min:<10.2f} {r_max:<10.2f} {r_mean:<10.2f} {r_median:<12.2f} {r_std:<10.2f}")
    print(f"{'Dist Streams':<18} {s_min:<10.2f} {s_max:<10.2f} {s_mean:<10.2f} {s_median:<12.2f} {s_std:<10.2f}")
    print("-" * 75)
    print(f"District Valid Cells: {int(mask.sum()):,}")
    print(f"Roads NaN in District: {int(np.isnan(dist_roads_valid).sum())}")
    print(f"Streams NaN in District: {int(np.isnan(dist_streams_valid).sum())}")
    print(f"Roads Outside All NaN: {bool(np.isnan(dist_roads_masked[~mask]).all())}")
    print(f"Streams Outside All NaN: {bool(np.isnan(dist_streams_masked[~mask]).all())}")
    print("=" * 75)

    # Verifications
    assert dist_roads_masked.shape == (H, W), f"dist_roads shape {dist_roads_masked.shape} != {(H, W)}"
    assert dist_streams_masked.shape == (H, W), f"dist_streams shape {dist_streams_masked.shape} != {(H, W)}"
    assert np.isnan(dist_roads_valid).sum() == 0, "Found NaNs inside district for dist_roads"
    assert np.isnan(dist_streams_valid).sum() == 0, "Found NaNs inside district for dist_streams"
    assert np.isnan(dist_roads_masked[~mask]).all(), "Non-NaN pixels found outside district mask for dist_roads"
    assert np.isnan(dist_streams_masked[~mask]).all(), "Non-NaN pixels found outside district mask for dist_streams"
    assert r_min == 0.0, f"Expected min road distance 0.0m, got {r_min}"
    assert s_min == 0.0, f"Expected min stream distance 0.0m, got {s_min}"
    assert 5000.0 < r_max < 15000.0, f"Road max distance out of realistic range: {r_max}m"
    assert 2000.0 < s_max < 10000.0, f"Stream max distance out of realistic range: {s_max}m"
    print("ALL PROXIMITY VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
    return dist_roads_masked, dist_streams_masked


if __name__ == "__main__":
    generate_proximity_rasters()
