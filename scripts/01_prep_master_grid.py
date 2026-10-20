"""01_prep_master_grid.py - Master Grid Setup and Copernicus DEM Reprojection.

1. Defines master metric grid aligned to Wayanad boundary bounding box (UTM Zone 43N, EPSG:32643, 30m resolution).
2. Rasterizes district boundary into binary mask (1 inside, 0 outside).
3. Reprojects Copernicus GLO-30 DEM from EPSG:4326 to EPSG:32643 onto master grid.
4. Outputs:
   - data/processed/master/grid.json
   - data/processed/master/mask.tif
   - data/processed/master/dem_30m.tif
   - data/processed/district.gpkg
"""
import json
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.warp import Resampling

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    RAW, PROC, MASTER_DIR, CRS, RES,
    get_master_grid_params, save_tif, to_grid
)


def prep_master_grid():
    boundary_path = RAW / "boundary" / "wayanad_boundary.geojson"
    if not boundary_path.exists():
        raise FileNotFoundError(f"District boundary not found at {boundary_path}")

    print(f"Loading district boundary from {boundary_path}...")
    boundary_gdf = gpd.read_file(boundary_path)
    dist = boundary_gdf.to_crs(CRS)

    # 1. Compute grid bounds snapped to 30.0m multiples
    grid_params = get_master_grid_params(dist, res=RES)
    T = grid_params["transform"]
    H = grid_params["height"]
    W = grid_params["width"]
    bounds = grid_params["bounds"]
    minx, miny, maxx, maxy = bounds

    print(f"Master Grid dimensions: {H} rows x {W} cols (total cells: {H * W:,})")
    print(f"Resolution: {RES:.1f} m x {RES:.1f} m")
    print(f"UTM Zone 43N Bounds: X [{minx:.1f}, {maxx:.1f}], Y [{miny:.1f}, {maxy:.1f}]")

    # Save grid metadata
    grid_dict = {
        "transform": list(T)[:6],
        "height": H,
        "width": W,
        "crs": CRS,
        "res": RES,
        "bounds": bounds,
    }
    grid_json_str = json.dumps(grid_dict, indent=2)
    (MASTER_DIR / "grid.json").write_text(grid_json_str)
    (PROC / "grid.json").write_text(grid_json_str)
    print(f"Saved master grid definition to {MASTER_DIR / 'grid.json'}")

    # Save projected district boundary
    dist_gpkg_path = PROC / "district.gpkg"
    dist.to_file(dist_gpkg_path, driver="GPKG")
    dist.to_file(MASTER_DIR / "district.gpkg", driver="GPKG")
    print(f"Saved projected district boundary to {dist_gpkg_path}")

    # 2. Rasterize boundary into mask (1 inside, 0 outside)
    mask = rasterize(
        [(geom, 1) for geom in dist.geometry],
        out_shape=(H, W),
        transform=T,
        fill=0,
        dtype="uint8",
    )
    save_tif(MASTER_DIR / "mask.tif", mask, T, nodata=0)
    save_tif(PROC / "mask.tif", mask, T, nodata=0)
    print(f"Saved mask raster to {MASTER_DIR / 'mask.tif'}")

    # 3. Reproject DEM to master grid
    dem_tiles = sorted((RAW / "dem").glob("*.tif"))
    if not dem_tiles:
        raise FileNotFoundError(f"No DEM GeoTIFF files found in {RAW / 'dem'}")

    dem = np.full((H, W), np.nan, dtype="float32")
    for p in dem_tiles:
        print(f"Reprojecting DEM tile {p.name} to master grid at {RES}m resolution...")
        tile = to_grid(p, T, H, W, resampling=Resampling.bilinear, dtype="float32", nodata=np.nan)
        dem = np.where(np.isnan(dem), tile, dem)

    # Save DEM in master and base processed directory
    save_tif(MASTER_DIR / "dem_30m.tif", dem, T, nodata=np.nan)
    save_tif(PROC / "dem.tif", dem, T, nodata=np.nan)
    print(f"Saved reprojected DEM to {MASTER_DIR / 'dem_30m.tif'}")

    # 4. Rigorous verification and summary
    valid_pixels = int(mask.sum())
    raster_area_sqkm = valid_pixels * (RES ** 2) / 1e6
    vector_area_sqkm = float(dist.area.sum() / 1e6)
    area_diff_pct = abs(raster_area_sqkm - vector_area_sqkm) / vector_area_sqkm * 100

    dem_district = dem[mask == 1]
    dem_nan_count = int(np.isnan(dem_district).sum())
    min_elev = float(np.nanmin(dem_district))
    max_elev = float(np.nanmax(dem_district))
    mean_elev = float(np.nanmean(dem_district))
    std_elev = float(np.nanstd(dem_district))

    print("\n" + "=" * 60)
    print("MASTER GRID VERIFICATION REPORT")
    print("=" * 60)
    print(f"Grid CRS:             {CRS}")
    print(f"Grid Shape:           {H} x {W} (Height x Width)")
    print(f"Cell Resolution:      {RES} m x {RES} m ({RES**2} m^2/pixel)")
    print(f"Valid District Cells: {valid_pixels:,} / {H * W:,} ({valid_pixels / (H * W) * 100:.2f}%)")
    print(f"Raster Area:          {raster_area_sqkm:.3f} km^2")
    print(f"Vector Area:          {vector_area_sqkm:.3f} km^2 (Discrepancy: {area_diff_pct:.4f}%)")
    print(f"DEM NaN in District:  {dem_nan_count}")
    print(f"Elevation Min:        {min_elev:.2f} m")
    print(f"Elevation Max:        {max_elev:.2f} m")
    print(f"Elevation Mean:       {mean_elev:.2f} m (Std: {std_elev:.2f} m)")
    print("=" * 60)

    # Assertions
    assert valid_pixels > 0, "District mask contains 0 valid pixels!"
    assert area_diff_pct < 0.5, f"Raster area discrepancy too large ({area_diff_pct:.2f}%)!"
    assert dem_nan_count == 0, f"Found {dem_nan_count} NaNs in DEM within district boundary!"
    assert 0 < min_elev < 500, f"Suspicious minimum elevation: {min_elev}"
    assert 1800 < max_elev < 3000, f"Suspicious maximum elevation: {max_elev}"
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    prep_master_grid()
