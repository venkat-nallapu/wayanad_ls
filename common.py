"""Shared configuration and geospatial helper utilities for Project 7B:
Landslide Susceptibility Mapping of Wayanad District, Kerala.
"""
from pathlib import Path
import json
import math
import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, Resampling

# Study area configuration
DISTRICT = "Wayanad"
CRS = "EPSG:32643"          # Projected UTM Zone 43N (metric)
RES = 30.0                  # Pixel resolution in meters
SEED = 42
TRAIN_FRAC = 0.70
NONSLIDE_MIN_DIST = 300     # Minimum distance in meters from any landslide for negative samples

# Directory layout
PROJECT_ROOT = Path(__file__).resolve().parent
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs"
REPORTS = OUT / "reports"

# Processed subdirectories
MASTER_DIR = PROC / "master"
FACTORS_CONT_DIR = PROC / "factors_continuous"
FACTORS_BIN_DIR = PROC / "factors_binned"
SAMPLES_DIR = PROC / "samples"
FINAL_DIR = PROC / "final"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"

# Create directories if they do not exist
for p in (RAW, PROC, OUT, REPORTS, MASTER_DIR, FACTORS_CONT_DIR, FACTORS_BIN_DIR, SAMPLES_DIR, FINAL_DIR, TABLES_DIR, FIGURES_DIR):
    p.mkdir(parents=True, exist_ok=True)


def get_master_grid_params(boundary_gdf, res: float = RES):
    """Calculate snapped bounding box and grid dimensions from a boundary GeoDataFrame.
    
    Snaps bounds outwards to clean multiples of `res` to guarantee reproducible alignment.
    """
    if boundary_gdf.crs != CRS:
        boundary_gdf = boundary_gdf.to_crs(CRS)
        
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    minx = math.floor(minx / res) * res
    miny = math.floor(miny / res) * res
    maxx = math.ceil(maxx / res) * res
    maxy = math.ceil(maxy / res) * res
    
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)  # North-up grid
    
    return {
        "bounds": [minx, miny, maxx, maxy],
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "width": width,
        "height": height,
        "transform": transform,
        "crs": CRS,
        "res": res,
    }


def load_grid():
    """Load master grid affine transform, height, and width."""
    candidates = [
        MASTER_DIR / "grid.json",
        PROC / "grid.json",
    ]
    for p in candidates:
        if p.exists():
            g = json.loads(p.read_text())
            return Affine(*g["transform"]), g["height"], g["width"]
    raise FileNotFoundError(f"Master grid definition not found. Looked in: {[str(c) for c in candidates]}")


def save_tif(name_or_path, arr, T, nodata=None, crs=CRS):
    """Save a 2D numpy array to GeoTIFF with Deflate compression.
    
    Accepts full Path, path relative to project root, or bare filename (which defaults to data/processed).
    """
    arr = np.asarray(arr)
    target = Path(name_or_path)
    if not target.suffix:
        target = target.with_suffix(".tif")
        
    if target.is_absolute():
        final_path = target
    else:
        # Check if the relative path already starts with project-level dirs (e.g. data/, outputs/)
        parts = target.parts
        if parts and parts[0] in ("data", "outputs", "reports"):
            final_path = PROJECT_ROOT / target
        elif parts and parts[0] in ("master", "factors_continuous", "factors_binned", "samples", "final"):
            final_path = PROC / target
        else:
            final_path = PROC / target

    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(
        final_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=T,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)
    return final_path


def load_tif(name_or_path):
    """Load raster band 1 as a 2D numpy array.
    
    Searches target path directly, relative to PROJECT_ROOT, then searches processed subdirectories.
    """
    target = Path(name_or_path)
    if not target.suffix:
        target = target.with_suffix(".tif")
        
    if target.is_absolute() and target.exists():
        chosen_path = target
    elif (PROJECT_ROOT / target).exists():
        chosen_path = PROJECT_ROOT / target
    elif (PROC / target).exists():
        chosen_path = PROC / target
    else:
        candidates = [
            MASTER_DIR / target.name,
            FACTORS_CONT_DIR / target.name,
            FACTORS_BIN_DIR / target.name,
            SAMPLES_DIR / target.name,
            FINAL_DIR / target.name,
            PROC / target.name,
        ]
        chosen_path = None
        for c in candidates:
            if c.exists():
                chosen_path = c
                break
        if chosen_path is None:
            raise FileNotFoundError(f"Cannot find raster file: {name_or_path}")
            
    with rasterio.open(chosen_path) as src:
        return src.read(1)


def to_grid(path, T, H, W, resampling=Resampling.bilinear,
            dtype="float32", nodata=np.nan, band=1):
    """Warp any raster onto the master grid (matching CRS, extent, and pixel resolution)."""
    src_path = Path(path)
    if not src_path.is_absolute() and not src_path.exists() and (PROJECT_ROOT / src_path).exists():
        src_path = PROJECT_ROOT / src_path

    # Handle default nodata when target dtype is integer
    if np.issubdtype(np.dtype(dtype), np.integer) and (nodata is None or (isinstance(nodata, float) and np.isnan(nodata))):
        fill_val = 0
        dst_nodata = 0
    else:
        fill_val = nodata
        dst_nodata = nodata

    dst = np.full((H, W), fill_val, dtype=dtype)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=T,
            dst_crs=CRS,
            resampling=resampling,
            dst_nodata=dst_nodata,
        )
    return dst
