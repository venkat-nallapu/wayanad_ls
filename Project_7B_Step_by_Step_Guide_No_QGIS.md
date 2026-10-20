# Project 7B: Landslide Susceptibility Map of One Kerala District
## Step-by-step guide using the Frequency Ratio (FR) method, Python only (no QGIS)

**Budget:** 28 hours (8 weeks at about 3.5 h/week). **Lab work:** none.

### How to read this guide

- Tags from your research draft (28 Sep 2026): **[V2]** verified in 2+ sources, **[V1]** one source, **[U]** unverified.
- **[Explainer]** marks standard GIS/statistics practice that I am explaining. It is not from the draft's sources.
- **[U: check]** marks things I could not verify (file names, URLs, column names). Confirm them when you download the data.
- Every number in a worked example is **made up for illustration**.

---

## Contents

0. [What changes without QGIS, and what stays the same](#0-what-changes-without-qgis-and-what-stays-the-same)
1. [Setup](#1-setup)
2. [Step 1: Scope, district and proposal](#step-1-scope-district-and-proposal)
3. [Step 2: Landslide inventory](#step-2-landslide-inventory)
4. [Step 3: District boundary and the master grid](#step-3-district-boundary-and-the-master-grid)
5. [Step 4: Terrain factors from the DEM](#step-4-terrain-factors-from-the-dem)
6. [Step 5: Land cover, rainfall, lithology](#step-5-land-cover-rainfall-lithology)
7. [Step 6: Distance to streams and roads](#step-6-distance-to-streams-and-roads)
8. [Step 7: Quality check of the raster stack](#step-7-quality-check-of-the-raster-stack)
9. [Step 8: Bin each factor into classes](#step-8-bin-each-factor-into-classes)
10. [Step 9: Split landslide points, create non-landslide points](#step-9-split-landslide-points-create-non-landslide-points)
11. [Step 10: Compute FR for every class](#step-10-compute-fr-for-every-class)
12. [Step 11: Build the susceptibility index (LSI)](#step-11-build-the-susceptibility-index-lsi)
13. [Step 12: Classify LSI into zones and make the map](#step-12-classify-lsi-into-zones-and-make-the-map)
14. [Step 13: Validate with ROC-AUC](#step-13-validate-with-roc-auc)
15. [Step 14: Quality checks and sanity tests](#step-14-quality-checks-and-sanity-tests)
16. [Step 15: Optional extras (sensitivity, spatial split, RF + SHAP)](#step-15-optional-extras)
17. [Step 16: Write the report](#step-16-write-the-report)
18. [28-hour plan](#28-hour-plan)
19. [Limitations, claims, risks, verify list](#limitations-claims-risks-and-verify-list)

---

## 0. What changes without QGIS, and what stays the same

**The method is identical.** FR, the susceptibility index (LSI), zoning and ROC-AUC do not depend on software. QGIS is only a way to do raster operations. Python does the same operations in code, and you gain a reproducible pipeline: change one setting, rerun everything.

### QGIS operation → Python equivalent

| QGIS step in the draft/PDF | Python replacement | Library |
|---|---|---|
| Load, reproject, clip, resample layers | `rasterio.warp.reproject` onto one master grid | rasterio |
| Slope, aspect (Raster > Analysis) | Horn 3x3 kernels (same algorithm family as GDAL's default) | scipy.ndimage |
| Curvature (SAGA/GRASS) | Laplacian of the DEM | scipy.ndimage |
| Rasterise geology polygons | `rasterio.features.rasterize` | rasterio |
| Proximity (raster distance) | `distance_transform_edt` | scipy.ndimage |
| Sample raster values at points | index arrays by row/column | numpy |
| Zonal histogram / unique values report | `np.bincount` | numpy |
| Reclassify by table | lookup array | numpy |
| Raster Calculator (add rasters) | `+` on numpy arrays | numpy |
| ROC-AUC | `roc_auc_score`, `roc_curve` | scikit-learn |
| Map layout | matplotlib | matplotlib |

### What "same outcome" means here **[Explainer]**

- You will produce the same deliverables: FR tables, an LSI raster (GeoTIFF), a zoned susceptibility map, success-rate and prediction-rate AUC, ROC curves, and zone statistics.
- Results will not be bit-identical to a QGIS run. Slope/curvature algorithms, resampling choices and binning differ slightly between tools. Expect small differences in FR values and AUC (typically a few hundredths). I have not tested this, so treat it as an expectation rather than a measurement **[U]**.
- To match QGIS closely, use Horn's method for slope (done below), the same bin edges, and the same random seed/split for both runs if you ever compare.
- Your outputs are standard GeoTIFFs. If you ever want to view them in QGIS for cosmetics, they open directly, but nothing in this guide requires it.

---

## 1. Setup

**Time: 0.5 h.** You already use WSL2 + uv, so:

```bash
uv init landslide-7b && cd landslide-7b
uv add numpy pandas geopandas rasterio scipy scikit-learn matplotlib shapely pyproj jenkspy
uv add shap                 # only for the optional SHAP step
uv add --dev jupyterlab     # optional, if you prefer notebooks
```

If you prefer Google Colab, use `pip install geopandas rasterio jenkspy shap` instead. (Colab already has numpy, scipy, scikit-learn, matplotlib.)

### Folder layout

```
landslide-7b/
  data/
    raw/
      boundary/      # district boundaries
      inventory/     # landslide points
      dem/           # Copernicus GLO-30 tiles (.tif)
      landcover/     # ESA WorldCover tiles (.tif)
      chirps/        # CHIRPS monthly rasters (.tif)
      geology/       # lithology polygons
      osm/           # waterways, roads
    processed/       # everything the scripts create
  outputs/           # maps, tables, figures
  common.py
  s03_grid.py  s04_terrain.py  s05_layers.py  s06_distance.py
  s07_qc.py    s08_bins.py     s09_split.py   s10_fr.py
  s11_lsi.py   s12_zones.py    s13_validate.py  s14_checks.py
```

Each script reads files from `data/processed` and writes new ones. Re-running one step never requires redoing earlier steps.

### `common.py` (shared settings and helpers)

```python
# common.py
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

DISTRICT = "Wayanad"        # or "Malappuram"
CRS = "EPSG:32643"          # UTM zone 43N; Kerala lies within it (about 74.9E to 77.4E)
RES = 30                    # metres
SEED = 42
TRAIN_FRAC = 0.70
NONSLIDE_MIN_DIST = 300     # m, non-landslide points must be this far from any landslide

RAW, PROC, OUT = Path("data/raw"), Path("data/processed"), Path("outputs")
for p in (RAW, PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)


def load_grid():
    g = json.loads((PROC / "grid.json").read_text())
    return Affine(*g["transform"]), g["height"], g["width"]


def save_tif(name, arr, T, nodata=None):
    arr = np.asarray(arr)
    with rasterio.open(
        PROC / f"{name}.tif", "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs=CRS, transform=T, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def load_tif(name):
    with rasterio.open(PROC / f"{name}.tif") as src:
        return src.read(1)


def to_grid(path, T, H, W, resampling=Resampling.bilinear,
            dtype="float32", nodata=np.nan, band=1):
    """Warp any raster onto the master grid (same CRS, extent, pixel size)."""
    dst = np.full((H, W), nodata, dtype=dtype)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=T, dst_crs=CRS,
            resampling=resampling, dst_nodata=nodata,
        )
    return dst
```

---

## Step 1: Scope, district and proposal

**Time: about 1 h. Week 1.**

**Purpose.** Fix what you are building before touching data, and get your professor's approval.

**What to do**
1. Pick **one** district. The draft's examples are **Wayanad** and **Malappuram**. In the Landslide Atlas of India, Kerala districts rank: Thrissur 3rd, Palakkad 5th, Malappuram 7th, Kozhikode 10th, Wayanad 13th **[V2]**.
2. Choose by inventory size, not only by fame. A district with too few landslide points gives unstable FR values (see Step 2).
3. Write a one-page proposal:
   - **Problem.** About 12.6% of India's land (0.42 million sq km) is landslide-prone **[V1]** (check the NDMA NLRMP 2025 document). The Landslide Atlas lists about 80,000 landslides in 1998-2022 **[V2]**. The Wayanad disaster of 30 July 2024 left over 400 dead or missing; sources differ (Wikipedia 420 deaths; Sci Rep 2025 400 dead and 200 missing) **[V2, range]**.
   - **Objective.** A susceptibility map of the district with a validated AUC.
   - **Method.** FR with about 6 factors, 70/30 split, ROC-AUC.
   - **Scope statement.** It is a **susceptibility** map, not a hazard or risk map.
4. Plain-language pitch for the professor **[Explainer]**: *Heavy monsoon rain on steep, weathered slopes can trigger landslides that destroy homes and roads. Using past landslide locations plus free terrain, rainfall and land-cover data, this project maps where in one district the same conditions occur, so planners know which areas deserve closer investigation.*

**Concepts to be clear on [Explainer]**
- **Susceptibility** = *where* a landslide is likely, given terrain and ground conditions.
- **Hazard** adds *when* and *how big* (rainfall return period, runout).
- **Risk** adds what is exposed (people, buildings, roads).
- The assumption behind FR: future landslides are more likely under the same conditions that produced past landslides.

**Deliverable:** the 1-page proposal.

---

## Step 2: Landslide inventory

**Time: about 1.5 h. Week 1.** Register for portals early because logins can be slow.

**Purpose.** The inventory is the "ground truth". Its quality limits everything downstream.

**Sources (from the draft)**
- GSI Bhukosh (registration needed).
- The bharatlas mirror (30,842 points, CC0) **[V1]** as a fallback.
- The draft notes that Kerala inventories are coarse and that an automated inventory is not available for the state (PMC12749052). Record this in your limitations.

**What to do**
1. Download the inventory and put it in `data/raw/inventory/`.
2. Load it and count the points inside your district:

```python
import geopandas as gpd, pandas as pd
from common import *

# GeoJSON / shapefile:
inv = gpd.read_file(RAW / "inventory/landslides.geojson")
# or a CSV with lat/lon columns (column names [U: check]):
# df = pd.read_csv(RAW / "inventory/landslides.csv")
# inv = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

if inv.crs is None:
    inv = inv.set_crs("EPSG:4326")          # only if you are sure it is lat/lon
print(inv.crs, len(inv))
print(inv.geom_type.value_counts())          # points or polygons?
```

3. If the data are polygons, use a representative point, ideally the **initiation (crown/source) point**, not the centroid of the runout. The point should represent where failure began. **[Explainer]**
4. Count the points inside the district (after Step 3 gives you the boundary). A rough rule of thumb is a few hundred points. Fewer makes many classes unstable. If you have too few, widen to a neighbouring district or merge another source. **[Explainer]**

**Check:** plot the points on top of a road layer later (Step 14). Landslides clustered along roads show an observation bias.

**Common mistake:** treating a lat/lon CSV as if it were already projected. Always set `crs="EPSG:4326"` first, then `to_crs`.

---

## Step 3: District boundary and the master grid

**Time: about 1 h. Week 2.** **This step replaces the "same CRS, same extent" fiddling you would do in QGIS.**

**Purpose.** Define **one grid** (CRS, pixel size, extent, rows and columns). Every factor will be warped onto this grid. After that, pixel (row, col) refers to the same ground location in every layer, so there is no alignment problem.

**Boundary source.** The draft does not name one. Candidates: Survey of India / Bhukosh district boundaries, or open datasets such as DataMeet or GADM **[U: check availability]**. Put it in `data/raw/boundary/`.

```python
# s03_grid.py
import json, math
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from common import *

dist_all = gpd.read_file(RAW / "boundary/districts.shp")      # name [U: check]
print(dist_all.columns)                                        # find the district-name column
col = "DISTRICT"                                               # edit to match
dist = dist_all[dist_all[col].str.lower() == DISTRICT.lower()].to_crs(CRS)
assert len(dist) == 1, "district not found or duplicated"

minx, miny, maxx, maxy = dist.total_bounds
minx = math.floor(minx / RES) * RES
miny = math.floor(miny / RES) * RES
maxx = math.ceil(maxx / RES) * RES
maxy = math.ceil(maxy / RES) * RES
W = int((maxx - minx) / RES)
H = int((maxy - miny) / RES)
T = from_origin(minx, maxy, RES, RES)         # north-up grid

(PROC / "grid.json").write_text(json.dumps(
    {"transform": list(T)[:6], "height": H, "width": W, "crs": CRS, "res": RES}))

mask = rasterize([(g, 1) for g in dist.geometry], out_shape=(H, W),
                 transform=T, fill=0, dtype="uint8")
save_tif("mask", mask, T, nodata=0)
dist.to_file(PROC / "district.gpkg", driver="GPKG")
print(f"grid {H} x {W} px, district pixels: {int(mask.sum()):,}")
```

**Check:** pixel count × 900 m² should roughly equal the district's official area. A big mismatch means a CRS error.

**Why snap to multiples of 30 m:** it keeps the grid clean and makes the extent reproducible.

---

## Step 4: Terrain factors from the DEM

**Time: about 3 h. Week 2.**

**Purpose.** Derive slope, aspect and curvature from the DEM.

**Data (from the draft):** Copernicus GLO-30 DEM via OpenTopography (login needed). Put the downloaded tiles in `data/raw/dem/`.

**Optional no-login route [U: check].** GLO-30 is also published on an AWS open-data bucket as cloud-optimised GeoTIFFs. Tile names look like `Copernicus_DSM_COG_10_N11_00_E076_00_DEM.tif`, and rasterio can read them over `/vsicurl/`. Verify the exact bucket URL and naming before relying on it. You will probably need two tiles for Wayanad (around 11-12°N, 75-76°E), so check the tile coverage against your district's bounds.

### Why you must compute slope on the projected DEM

The raw DEM is in degrees of latitude/longitude. Slope needs horizontal distance in metres. So first warp the DEM onto the master grid (UTM, 30 m), then compute slope on that.

```python
# s04_terrain.py
import numpy as np
from scipy.ndimage import correlate, laplace
from common import *

T, H, W = load_grid()

# 1) Mosaic DEM tiles onto the master grid
dem = np.full((H, W), np.nan, dtype="float32")
for p in sorted((RAW / "dem").glob("*.tif")):
    tile = to_grid(p, T, H, W)                  # bilinear
    dem = np.where(np.isnan(dem), tile, dem)
save_tif("dem", dem, T, nodata=np.nan)

# 2) Slope and aspect with Horn's 3x3 method
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * RES)
ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * RES)
dzdx = correlate(dem, kx, mode="nearest")        # rises toward east: positive
dzdy = correlate(dem, ky, mode="nearest")        # rises toward SOUTH (down the rows): positive

slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")

# Aspect = compass bearing the slope FACES (downslope direction), 0 = north, clockwise
aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360).astype("float32")
aspect[slope < 1] = -1                           # nearly flat: no aspect

# 3) Curvature: Laplacian (total curvature). Positive = concave/valley-like here.
curv = (laplace(dem) / RES**2).astype("float32")

save_tif("slope", slope, T, nodata=np.nan)
save_tif("aspect", aspect, T, nodata=np.nan)
save_tif("curv", curv, T, nodata=np.nan)
```

### Sanity-test the aspect formula (5 minutes, do not skip)

Aspect sign errors are the classic bug. Test on a synthetic tilted plane:

```python
import numpy as np
from scipy.ndimage import correlate
RES = 30
x = np.arange(50)[None, :] * RES            # east increases with column
plane = 100 - 0.2 * x + 0 * np.arange(50)[:, None]    # drops toward EAST
kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) / (8*RES)
ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) / (8*RES)
dzdx, dzdy = correlate(plane, kx, mode="nearest"), correlate(plane, ky, mode="nearest")
asp = np.degrees(np.arctan2(-dzdx, dzdy)) % 360
print(asp[25, 25])     # expect about 90 (east-facing)
# Repeat with a plane that drops toward the north (decreasing row index): expect 0.
```

### Notes **[Explainer]**
- **Curvature convention differs between software.** Some tools flip the sign, and some separate profile and plan curvature. The Laplacian here is a single, simple measure. Whatever you use, state it in the report and keep the sign convention consistent in your interpretation.
- **Edge pixels** of the bounding box are slightly less accurate. They sit outside or on the district edge, so the effect is small.
- **Elevation itself** is an optional extra factor, but it is not in the draft's list. Skip it unless your professor asks.

**Check:** print `np.nanmin/max` of slope. Values above about 70° or below 0 mean a bug or DEM noise. Look at a quick `plt.imshow(slope)` to confirm that steep terrain looks like mountains, not stripes.

---

## Step 5: Land cover, rainfall, lithology

**Time: about 3 h. Week 3.**

### 5a. Land cover (ESA WorldCover 10 m, no login, from the draft)

Resample **categorical** data with `mode` or `nearest`, never `bilinear` (averaging class codes is meaningless).

```python
# s05_layers.py (part 1)
import numpy as np
from rasterio.warp import Resampling
from common import *

T, H, W = load_grid()
lc = np.zeros((H, W), dtype="uint8")
for p in sorted((RAW / "landcover").glob("*.tif")):
    tile = to_grid(p, T, H, W, resampling=Resampling.mode, dtype="uint8", nodata=0)
    lc = np.where(lc == 0, tile, lc).astype("uint8")
save_tif("landcover", lc, T, nodata=0)
```

WorldCover class codes **[U: check against the product's legend]**: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare/sparse, 70 snow/ice, 80 water, 90 wetland, 95 mangroves, 100 moss/lichen.

> **Correction to the earlier PDF.** The worked LSI example used "Plantation" as a land-cover class. WorldCover does not separate plantations from natural forest (both are mostly "tree cover"). That was an illustration only. Real class names will come from the legend above.

### 5b. Rainfall (CHIRPS, from the draft)

CHIRPS is about 0.05° (roughly 5 km). Within one district this gives only a handful of distinct values, so it separates areas weakly. State this as a limitation. Resampling to 30 m with bilinear interpolation makes a smooth surface but adds **no new information**.

Choose one summary and justify it: **mean monsoon (June-September) total** or mean annual total. For Kerala, monsoon rainfall is the more landslide-relevant choice **[Explainer]**.

```python
# s05_layers.py (part 2)
from collections import defaultdict
by_year = defaultdict(list)
for p in sorted((RAW / "chirps").glob("*.tif")):
    yr, mo = p.stem.split(".")[-2:]        # file names like chirps-v2.0.2020.06.tif [U: check]
    if mo in {"06", "07", "08", "09"}:
        arr = to_grid(p, T, H, W)          # bilinear
        arr[arr < 0] = np.nan              # CHIRPS no-data is negative
        by_year[yr].append(arr)

rain = np.mean([np.sum(v, axis=0) for v in by_year.values()], axis=0).astype("float32")
save_tif("rain", rain, T, nodata=np.nan)
print("years used:", sorted(by_year), " rain range (mm):", np.nanmin(rain), np.nanmax(rain))
```

Use a long, fixed period (e.g. 10-20 years) and say which in the report.

### 5c. Lithology (the draft lists the factor but gives no data source)

- **Candidate:** GSI geology layers (Bhukosh or district geology maps) **[U: check availability]**.
- **Fallback:** a coarse global lithology map (e.g. GLiM) **[U: check]**, which is likely too coarse for one district.
- **If you find nothing usable:** drop to **5 factors** and say why. The draft's minimum viable plan already assumes this.

```python
# s05_layers.py (part 3)
import geopandas as gpd
from rasterio.features import rasterize

geo = gpd.read_file(RAW / "geology/geology.shp").to_crs(CRS)      # name [U: check]
print(geo.columns)
LITH_COL = "LITHO"                                                  # edit to match
codes = {n: i + 1 for i, n in enumerate(sorted(geo[LITH_COL].dropna().unique()))}
shapes = ((g, codes[n]) for g, n in zip(geo.geometry, geo[LITH_COL]) if n in codes)
litho = rasterize(shapes, out_shape=(H, W), transform=T, fill=0, dtype="uint8")
save_tif("litho", litho, T, nodata=0)
import pandas as pd
pd.Series(codes, name="code").to_csv(OUT / "litho_codes.csv")       # keep for the report
```

Merge rare rock types into broader groups. Otherwise you get classes with 1-2 landslides.

---

## Step 6: Distance to streams and roads

**Time: about 1.5 h. Week 3.**

**Data (from the draft):** Geofabrik OSM (waterways and roads). The regional extract is a `.osm.pbf` **[U: check which Geofabrik extract covers Kerala]**. Read it with geopandas/pyogrio; the PBF exposes layers such as `lines` **[U: check]**.

```python
# s06_distance.py
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from common import *

T, H, W = load_grid()
PAD = 100                                    # pixels (3 km) so lines just outside the district count
Tp = Affine(T.a, 0, T.c - PAD * RES, 0, T.e, T.f + PAD * RES)
Hp, Wp = H + 2 * PAD, W + 2 * PAD

lines = gpd.read_file(RAW / "osm/india-south.osm.pbf", layer="lines")   # [U: check file/layer]
lines = lines.to_crs(CRS)
# 'other_tags' holds waterway/highway for lines in the default OGR schema [U: check]
# Easiest robust route: filter on the columns your file has, e.g.:
streams = lines[lines["waterway"].isin(["river", "stream", "canal"])] if "waterway" in lines else lines.iloc[0:0]
roads = lines[lines["highway"].notna()] if "highway" in lines else lines.iloc[0:0]

def dist_raster(gdf):
    burn = rasterize(((g, 1) for g in gdf.geometry), out_shape=(Hp, Wp),
                     transform=Tp, fill=0, dtype="uint8", all_touched=True)
    d = distance_transform_edt(burn == 0) * RES          # metres to nearest line
    return d[PAD:-PAD, PAD:-PAD].astype("float32")

save_tif("d_stream", dist_raster(streams), T, nodata=np.nan)
save_tif("d_road", dist_raster(roads), T, nodata=np.nan)
```

**Decisions to write down [Explainer]:** which OSM tags count as "stream" and "road" (e.g. exclude footpaths if you like). Change them consistently if you rerun.

**Check:** `plt.imshow(d_stream)` should show dark lines along rivers. If the whole raster is one value, the filter returned nothing (the `iloc[0:0]` fallback above hides that, so print `len(streams)`, `len(roads)`).

**Why the padding:** without it, pixels near the district edge would measure distance only to lines inside the grid, which biases them upward.

---

## Step 7: Quality check of the raster stack

**Time: about 0.5 h. Week 3.**

**Purpose.** Prove that all layers share a grid and define one **VALID** mask: pixels inside the district where **every** factor has data. Use this mask for everything, so area percentages and landslide counts refer to the same pixels.

```python
# s07_qc.py
import numpy as np
from common import *

T, H, W = load_grid()
names = ["slope", "aspect", "curv", "litho", "landcover", "rain", "d_stream", "d_road"]
layers = {n: load_tif(n) for n in names}
for n, a in layers.items():
    assert a.shape == (H, W), f"{n} shape mismatch {a.shape}"
    print(f"{n:10s} dtype={a.dtype} min={np.nanmin(a):.2f} max={np.nanmax(a):.2f}")

mask = load_tif("mask").astype(bool)
valid = mask.copy()
for n in ["slope", "aspect", "curv", "rain", "d_stream", "d_road"]:
    valid &= np.isfinite(layers[n])
for n in ["litho", "landcover"]:
    valid &= layers[n] > 0
print("district pixels:", mask.sum(), " valid pixels:", valid.sum(),
      f" ({100 * valid.sum() / mask.sum():.1f}% kept)")
save_tif("valid", valid.astype("uint8"), T, nodata=0)
```

**Interpret.** If much less than about 95% is kept, find which layer has holes (most often geology polygons that do not cover the whole district, or land cover over water). Fill or restrict the district rather than letting a factor silently shrink the study area.

---

## Step 8: Bin each factor into classes

**Time: about 2 h. Week 4.**

**Purpose.** FR works on **classes**. Categorical factors already have them. Continuous factors need bins.

**Guidelines [Explainer]**
- Aim for about **5-10 classes** per factor.
- Use equal-interval bins where the numbers are meaningful (slope in 5° steps), quantiles for skewed data, and natural breaks (Jenks) as an alternative.
- Avoid classes so small that only a few landslide points can fall in them.
- Aspect: 8 compass directions plus "flat".
- Write your bin edges in a table. You will report them, and you can test sensitivity later.

```python
# s08_bins.py
import numpy as np, pandas as pd
from common import *

T, H, W = load_grid()
valid = load_tif("valid").astype(bool)

def to_classes(arr, edges):
    """Bins are [e_i, e_{i+1}). Classes 1..n; 0 = no data."""
    cls = np.digitize(arr, edges[1:-1]) + 1
    cls[~np.isfinite(arr)] = 0
    return cls.astype("uint8")

BINS = {                                   # edit after looking at the histograms
    "slope":    [0, 5, 10, 15, 20, 25, 30, 35, 40, 90],
    "d_stream": [0, 100, 200, 300, 500, 1000, np.inf],
    "d_road":   [0, 100, 200, 300, 500, 1000, np.inf],
}
# Skewed or unit-less factors: quantile bins (6 classes)
for name, k in [("curv", 6), ("rain", 6)]:
    a = load_tif(name)
    edges = np.unique(np.nanquantile(a[valid], np.linspace(0, 1, k + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    BINS[name] = list(edges)

classes = {}
for name, edges in BINS.items():
    classes[name] = to_classes(load_tif(name), np.array(edges, dtype=float))

# Aspect: 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 9=flat
asp = load_tif("aspect")
a8 = (((np.where(np.isfinite(asp), asp, 0) + 22.5) % 360) // 45).astype(int) + 1
a8[asp < 0] = 9
a8[~np.isfinite(asp)] = 0
classes["aspect"] = a8.astype("uint8")

classes["litho"] = load_tif("litho")           # already categorical
classes["landcover"] = load_tif("landcover")

# Apply the VALID mask to every class raster (0 = excluded)
for name in classes:
    classes[name] = np.where(valid, classes[name], 0).astype("uint8")
    save_tif(f"cls_{name}", classes[name], T, nodata=0)

pd.DataFrame({k: [list(map(float, v))] for k, v in BINS.items()}).T.to_csv(OUT / "bin_edges.csv")
```

**Check:** for each factor, print the class pixel counts (`np.bincount`). If a class has a tiny share of the district, merge it into a neighbour now. Remember that bins are left-closed, `[e_i, e_{i+1})`.

**Recommended factor set for the budget (6 factors):** slope, lithology, land cover, rainfall, distance to streams, distance to roads. Add aspect and curvature if time allows (the draft lists 6-8).

---

## Step 9: Split landslide points, create non-landslide points

**Time: about 1.5 h. Week 4.**

**Purpose.** FR must be computed from **training** points only. **Validation** points stay hidden until Step 13. The ROC test also needs **non-landslide** points.

**Rules [Explainer]**
- Convert every point to a (row, col) pixel and keep **one point per pixel**. Two points in one pixel would double-count.
- Randomly assign about 70% training / 30% validation. Record the seed.
- Non-landslide points: random pixels at least `NONSLIDE_MIN_DIST` metres from **any** landslide, equal in number to the landslides (so train positives pair with train negatives and validation positives with validation negatives).

```python
# s09_split.py
import numpy as np
import geopandas as gpd
import rasterio
from scipy.ndimage import distance_transform_edt
from common import *

T, H, W = load_grid()
valid = load_tif("valid").astype(bool)
dist = gpd.read_file(PROC / "district.gpkg")

inv = gpd.read_file(RAW / "inventory/landslides.geojson").to_crs(CRS)   # as in Step 2
inv = inv[inv.within(dist.geometry.union_all())]
# if polygons: inv["geometry"] = inv.geometry.representative_point()

rows, cols = rasterio.transform.rowcol(T, inv.geometry.x.values, inv.geometry.y.values)
rc = np.unique(np.c_[np.array(rows), np.array(cols)], axis=0)            # one point per pixel
ok = (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W)
rc = rc[ok]
rc = rc[valid[rc[:, 0], rc[:, 1]]]
print("usable landslide pixels:", len(rc))

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(rc))
n_tr = int(TRAIN_FRAC * len(rc))
train, val = rc[idx[:n_tr]], rc[idx[n_tr:]]

# Non-landslide pool, far from every landslide
slide = np.zeros((H, W), bool)
slide[rc[:, 0], rc[:, 1]] = True
far = (distance_transform_edt(~slide) * RES) > NONSLIDE_MIN_DIST
cand = np.argwhere(far & valid)
neg = cand[rng.choice(len(cand), size=len(rc), replace=False)]
neg_tr, neg_val = neg[:len(train)], neg[len(train):len(train) + len(val)]

np.savez(PROC / "points.npz", train=train, val=val, neg_tr=neg_tr, neg_val=neg_val)
print(len(train), "train /", len(val), "validation landslides; negatives:", len(neg_tr), len(neg_val))
```

**Why the seed matters:** your AUC depends a little on the random split. Step 15 repeats it with several seeds.

**Honest caveat [Explainer].** Landslides near each other share conditions, so a random split gives a somewhat optimistic AUC. A spatial split (Step 15) is stricter.

---

## Step 10: Compute FR for every class

**Time: about 1.5 h. Week 4.**

**The formula**

```
FR = (% of training landslide points in the class) / (% of district area in the class)
```

| FR | Meaning |
|---|---|
| = 1 | Landslides occur in proportion to area. The class gives no information. |
| > 1 | More landslides than area alone predicts: landslide-prone. |
| < 1 | Fewer than expected: relatively stable. |
| = 0 | No training landslides in the class. Usually a very small class or a data gap. |

**Worked example (illustrative, not real data).** 200 training landslides; slope classes:

| Slope class | % of area | Landslides | % of landslides | FR |
|---|---|---|---|---|
| 0-10° | 40% | 10 | 5% | 5/40 = 0.13 |
| 10-20° | 30% | 40 | 20% | 20/30 = 0.67 |
| 20-30° | 20% | 80 | 40% | 40/20 = 2.00 |
| 30-45° | 10% | 70 | 35% | 35/10 = 3.50 |

Steep slopes cover 10% of the area but hold 35% of landslides, so FR = 3.5.

```python
# s10_fr.py
import numpy as np, pandas as pd
from common import *

pts = np.load(PROC / "points.npz")
train = pts["train"]
factors = ["slope", "litho", "landcover", "rain", "d_stream", "d_road"]   # add "aspect", "curv" if used

def fr_table(cls, train_rc, name):
    n = int(cls.max()) + 1
    area = np.bincount(cls[cls > 0], minlength=n)
    hits = np.bincount(cls[train_rc[:, 0], train_rc[:, 1]], minlength=n)
    df = pd.DataFrame({"bin": np.arange(n), "pixels": area, "slides": hits})
    df = df[(df.bin > 0) & (df.pixels > 0)].copy()
    df["area_pct"] = 100 * df.pixels / df.pixels.sum()
    df["slide_pct"] = 100 * df.slides / df.slides.sum()
    df["FR"] = df.slide_pct / df.area_pct
    df.insert(0, "factor", name)
    return df

tables = []
for f in factors:
    cls = load_tif(f"cls_{f}")
    t = fr_table(cls, train, f)
    assert abs(t.area_pct.sum() - 100) < 1e-6 and abs(t.slide_pct.sum() - 100) < 1e-6
    tables.append(t)

fr_all = pd.concat(tables, ignore_index=True)
fr_all.to_csv(OUT / "fr_tables.csv", index=False)
print(fr_all.round(2).to_string(index=False))
```

The assertions are the checks the PDF mentions: both percentage columns must total 100%. If they do not, you have a no-data or alignment problem.

**Handling awkward classes [Explainer]**
- **Zero FR.** Merge the class with a neighbour (go back to Step 8), or report it as "no landslides in the training set". A hard zero says a condition is impossible, which is overconfident.
- **Tiny classes (1-2 landslides).** Their FR is noisy. Merge them.
- **Plain FR gives every factor equal weight.** Some studies normalise the FR values or weight factors with AHP or statistical models. The draft lists hybrid AHP+ML as future work in the literature. You can mention it without implementing it.

**Deliverable:** `fr_tables.csv`. In the report, plot FR per class for each factor (Step 14).

---

## Step 11: Build the susceptibility index (LSI)

**Time: about 1 h. Week 5.**

**Idea.** Each pixel sits in one class per factor. Look up that class's FR and add them:

```
LSI(pixel) = FR(slope) + FR(litho) + FR(landcover) + FR(rain) + FR(d_stream) + FR(d_road)
```

**One-pixel example (illustrative)**

| Factor | Class this pixel falls in | FR |
|---|---|---|
| Slope | 30-45° | 3.50 |
| Lithology | weathered rock class | 1.80 |
| Rainfall | highest band | 1.40 |
| Land cover | tree cover | 0.90 |
| Distance to stream | 100-200 m | 1.20 |
| Distance to road | > 500 m | 0.70 |
| **LSI** | | **9.50** |

```python
# s11_lsi.py
import numpy as np, pandas as pd
from common import *

T, H, W = load_grid()
valid = load_tif("valid").astype(bool)
fr_all = pd.read_csv(OUT / "fr_tables.csv")

lsi = np.zeros((H, W), dtype="float32")
for f, t in fr_all.groupby("factor"):
    cls = load_tif(f"cls_{f}")
    lut = np.zeros(int(cls.max()) + 1, dtype="float32")        # lookup: class -> FR
    lut[t["bin"].astype(int).values] = t["FR"].values
    lsi += lut[cls]                                             # "Reclassify by table" + "Raster Calculator"

lsi[~valid] = np.nan
save_tif("lsi", lsi, T, nodata=np.nan)
print("LSI range:", np.nanmin(lsi), np.nanmax(lsi))
```

A higher LSI means more of the pixel's conditions are ones where landslides were over-represented. LSI is a **relative ranking**, not a probability.

---

## Step 12: Classify LSI into zones and make the map

**Time: about 2 h. Week 5.**

**Zoning options [Explainer]:** natural breaks (Jenks), quantiles, or equal interval. The same LSI looks different under each, so state which you used and why. Quantiles are easiest to defend and compare. Jenks fits the data's natural groupings.

```python
# s12_zones.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
from common import *

T, H, W = load_grid()
lsi = load_tif("lsi")
ok = np.isfinite(lsi)

METHOD = "quantile"                     # or "jenks"
if METHOD == "quantile":
    edges = np.nanquantile(lsi[ok], [0.2, 0.4, 0.6, 0.8])
else:
    import jenkspy
    edges = jenkspy.jenks_breaks(lsi[ok][::50], n_classes=5)[1:-1]   # subsample for speed

zone = np.digitize(lsi, edges) + 1              # 1..5
zone[~ok] = 0
zone = zone.astype("uint8")
save_tif("zones", zone, T, nodata=0)

LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]

# Map
dist = gpd.read_file(PROC / "district.gpkg")
minx, miny = T.c, T.f + H * T.e
extent = (minx, minx + W * RES, miny, miny + H * RES)
fig, ax = plt.subplots(figsize=(7, 8), dpi=200)
ax.imshow(np.ma.masked_equal(zone, 0), cmap=ListedColormap(COLORS), vmin=1, vmax=5, extent=extent)
dist.boundary.plot(ax=ax, color="k", lw=0.8)
for k, (lab, col) in enumerate(zip(LABELS, COLORS)):
    ax.plot([], [], "s", color=col, label=lab)
ax.legend(title="Susceptibility", loc="lower left", fontsize=8)
ax.set_title(f"{DISTRICT}: landslide susceptibility (FR, {METHOD} zones)")
ax.set_xlabel("Easting (m), UTM 43N"); ax.set_ylabel("Northing (m)")
ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.88), xycoords="axes fraction",
            ha="center", arrowprops=dict(arrowstyle="-|>"))
plt.tight_layout(); plt.savefig(OUT / "susceptibility_map.png"); plt.close()

# Zone statistics (area share only here; landslide share added in Step 13)
area_pct = 100 * np.bincount(zone[ok], minlength=6)[1:] / ok.sum()
pd.DataFrame({"zone": LABELS, "area_pct": area_pct.round(1)}).to_csv(OUT / "zone_area.csv", index=False)
```

**Check:** the map should look physically plausible: high susceptibility on steep, high-rainfall terrain, low on flat valley floors. If it looks like noise, review the FR trends (Step 14).

**Optional vector output:** `rasterio.features.shapes(zone, mask=zone > 0, transform=T)` converts zones to polygons, which can be saved as a GeoPackage.

---

## Step 13: Validate with ROC-AUC

**Time: about 2 h. Week 6.**

**What ROC-AUC does [Explainer].** Treat LSI as a classifier. At every threshold, compute the true positive rate (landslide points above the threshold) and the false positive rate (non-landslide points above it). Plot TPR against FPR. The area under that curve is the AUC.

| AUC | Rough interpretation |
|---|---|
| 0.5 | No better than random |
| 0.6-0.7 | Poor to fair |
| 0.7-0.8 | Acceptable to good |
| 0.8-0.9 | Good to very good (published FR/AHP studies: about 0.82-0.92) |
| > 0.9 | Excellent, but check for leakage or an easy non-landslide sample |

Published reference points from the draft **[V1 each]**: Malappuram AHP 0.897 on test data (PMC13534497); India-wide FR 0.905 (PMC12847720). Read both papers before quoting them.

**Report two AUCs**
- **Success rate** (training positives vs training negatives): how well the map fits the data used to build it.
- **Prediction rate** (validation positives vs validation negatives): ability to predict **unseen** landslides. **This is the one that matters.**

```python
# s13_validate.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from common import *

lsi = load_tif("lsi")
zone = load_tif("zones")
pts = np.load(PROC / "points.npz")
at = lambda rc: lsi[rc[:, 0], rc[:, 1]]

def evaluate(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[at(pos), at(neg)]
    return roc_auc_score(y, s), roc_curve(y, s)

auc_s, (fs, ts, _) = evaluate(pts["train"], pts["neg_tr"])
auc_p, (fp, tp, _) = evaluate(pts["val"], pts["neg_val"])
print(f"success rate AUC    = {auc_s:.3f}")
print(f"prediction rate AUC = {auc_p:.3f}")

fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.plot(fs, ts, label=f"Success rate (AUC {auc_s:.3f})")
ax.plot(fp, tp, label=f"Prediction rate (AUC {auc_p:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right"); ax.set_title("ROC curves")
plt.tight_layout(); plt.savefig(OUT / "roc.png"); plt.close()

# Sanity check: validation landslides per zone should rise from very low to very high
LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
area = np.bincount(zone[zone > 0], minlength=6)[1:]
hits = np.bincount(zone[pts["val"][:, 0], pts["val"][:, 1]], minlength=6)[1:]
tab = pd.DataFrame({"zone": LABELS,
                    "area_pct": 100 * area / area.sum(),
                    "val_slides_pct": 100 * hits / hits.sum()})
tab["density_ratio"] = tab.val_slides_pct / tab.area_pct       # like FR, per zone
tab.round(2).to_csv(OUT / "zone_validation.csv", index=False)
print(tab.round(2))
```

**How to read the results**
- A prediction-rate AUC close to the success rate means the map generalises. A large drop means overfitting or an unrepresentative split.
- `density_ratio` should rise from "Very low" to "Very high". If it does not, the zoning or factor set needs rethinking.

**What a good AUC does not prove.** It shows ranking performance on your test points. It does not prove the map is right everywhere, and it depends on how the non-landslide points were chosen.

---

## Step 14: Quality checks and sanity tests

**Time: about 1.5 h. Week 6.**

| Check | What to do | Why |
|---|---|---|
| Factor correlation | Pearson correlation and VIF between factors. A common flag is \|r\| > 0.7 or VIF > 5-10. | FR adds factors as if independent; correlated ones double-count. |
| Class sizes | Table of landslide count per class. Merge classes with very few points. | A class with 1-2 points gives a noisy FR. |
| Zero FR | Merge with a neighbour, or report as "none in training set". | A hard zero is overconfident. |
| Alignment | Plot points over the slope raster. | Misalignment silently gives wrong class lookups. |
| Inventory bias | Plot points over the road layer. Clustered along roads? | Easy-to-see landslides are over-recorded and can inflate the "distance to road" signal. |
| Sensible FR trends | Does FR rise with slope? Fall far from streams? | Reality check against mechanics. |

```python
# s14_checks.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import *

valid = load_tif("valid").astype(bool)
rng = np.random.default_rng(SEED)

# --- correlation and VIF on a random pixel sample (continuous factors only)
cont = ["slope", "curv", "rain", "d_stream", "d_road"]
rows, cols = np.nonzero(valid)
pick = rng.choice(len(rows), size=min(30000, len(rows)), replace=False)
X = pd.DataFrame({n: load_tif(n)[rows[pick], cols[pick]] for n in cont})
corr = X.corr()
vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=cont, name="VIF")
print(corr.round(2)); print(vif.round(2))
corr.to_csv(OUT / "correlation.csv"); vif.to_csv(OUT / "vif.csv")

# --- FR per class, one panel per factor (the "do the trends make sense?" figure)
fr_all = pd.read_csv(OUT / "fr_tables.csv")
fs = fr_all.factor.unique()
fig, axes = plt.subplots(2, (len(fs) + 1) // 2, figsize=(11, 6), dpi=150)
for ax, f in zip(axes.ravel(), fs):
    t = fr_all[fr_all.factor == f]
    ax.bar(t["bin"].astype(str), t["FR"], color=np.where(t["FR"] < 1, "#9db4d6", "#c0504d"))
    ax.axhline(1, color="k", ls="--", lw=0.8); ax.set_title(f, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "fr_by_factor.png"); plt.close()

# --- landslides per class (small-class check)
print(fr_all[fr_all.slides < 5][["factor", "bin", "pixels", "slides", "FR"]])
```

Categorical factors (lithology, land cover) are not covered by Pearson correlation. If you want a check, use Cramér's V between pairs, or just discuss it qualitatively.

---

## Step 15: Optional extras

**Time: about 1-4 h, only if time remains. Week 6.** Keep **FR as the main contribution** and drop extras first when you run short.

### 15a. Sensitivity to binning (about 1 h)

Rerun Steps 8, 10-13 with a different binning (e.g. quantiles instead of equal interval) and compare the prediction AUC. Writing the steps as scripts makes this a small edit to `BINS`.

### 15b. Factor-drop test (about 1 h)

Remove one factor at a time and see how the prediction AUC changes. This shows which factors matter most, which plain FR cannot say.

```python
import numpy as np
from sklearn.metrics import roc_auc_score
# per-factor FR rasters, built as in Step 11 (keep each lut[cls] in a dict: fr_maps[f])
base_pred = {f: fr_maps[f] for f in factors}
full = sum(base_pred.values())
at = lambda a, rc: a[rc[:, 0], rc[:, 1]]
y = np.r_[np.ones(len(val)), np.zeros(len(neg_val))]
for f in factors:
    lsi_minus = full - base_pred[f]
    s = np.r_[at(lsi_minus, val), at(lsi_minus, neg_val)]
    print(f, "AUC without it:", round(roc_auc_score(y, s), 3))
```

A big AUC drop means the factor is important.

### 15c. Repeat the split with several seeds (about 0.5 h)

Loop `SEED` over, say, 5 values, rerun Steps 9-13, and report the **mean and range** of the prediction AUC.

### 15d. Spatial split (about 1 h), a stricter validation

Train on the western part of the district and test on the eastern part (or use spatial blocks). This reduces the optimism of a random split.

```python
# in s09_split.py, replace the random split:
order = np.argsort(rc[:, 1])                  # sort landslides west -> east by column
n_tr = int(TRAIN_FRAC * len(rc))
train, val = rc[order[:n_tr]], rc[order[n_tr:]]
```

Use the same idea for the non-landslide points. Expect a lower AUC than the random split. That is an honest result.

### 15e. SHAP on the factors (about 3-4 h)

**FR is not a trained model, so SHAP needs a model.** Train a Random Forest on the same factors, compare it with FR, and use SHAP to rank factor importance and see effect shapes. It also supports the "hybrid AHP+ML" gap in the draft.

Two options **[Explainer]**:
1. **Recommended:** train RF (or XGBoost) on **raw** factor values at landslide/non-landslide points, same 70/30 split and same AUC, then run SHAP.
2. SHAP on the FR index itself adds nothing. LSI is a sum, so each factor's contribution is just its FR value minus the average. A bar chart of mean FR contribution per factor gives the same information.

```python
# s15_shap.py
import numpy as np, pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from common import *

pts = np.load(PROC / "points.npz")
raw = {n: load_tif(n) for n in ["slope", "rain", "d_stream", "d_road", "litho", "landcover", "aspect", "curv"]}

def sample(rc):
    d = {n: raw[n][rc[:, 0], rc[:, 1]] for n in ["slope", "rain", "d_stream", "d_road", "curv", "litho", "landcover"]}
    a = np.deg2rad(raw["aspect"][rc[:, 0], rc[:, 1]])
    flat = raw["aspect"][rc[:, 0], rc[:, 1]] < 0
    d["asp_sin"] = np.where(flat, 0, np.sin(a))        # circular encoding of aspect
    d["asp_cos"] = np.where(flat, 0, np.cos(a))
    return pd.DataFrame(d)

def make(pos, neg):
    X = pd.concat([sample(pos), sample(neg)], ignore_index=True)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return X, y

X_tr, y_tr = make(pts["train"], pts["neg_tr"])
X_va, y_va = make(pts["val"], pts["neg_val"])
X_all = pd.get_dummies(pd.concat([X_tr, X_va]), columns=["litho", "landcover"]).astype(float)
X_tr, X_va = X_all.iloc[:len(X_tr)], X_all.iloc[len(X_tr):]

rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
print("RF prediction-rate AUC:", round(roc_auc_score(y_va, rf.predict_proba(X_va)[:, 1]), 3))

sv = shap.TreeExplainer(rf).shap_values(X_va)
sv1 = sv[1] if isinstance(sv, list) else sv[..., 1]       # class 1 = landslide; shape varies by SHAP version

# Sum one-hot columns back into their original factor
group = lambda c: c.split("_")[0] if c.startswith(("litho_", "landcover_")) else c
cols = pd.Series(X_va.columns)
imp = pd.DataFrame(sv1, columns=X_va.columns).T.groupby(group).sum().T.abs().mean().sort_values()
imp.plot.barh(); plt.xlabel("mean |SHAP|"); plt.tight_layout(); plt.savefig(OUT / "shap_bar.png")
shap.summary_plot(sv1, X_va, show=False); plt.savefig(OUT / "shap_beeswarm.png", bbox_inches="tight")
```

**Cautions for SHAP**
- **Correlated factors** (e.g. slope vs distance to stream) split credit unpredictably. Run the correlation check first and say so in the report.
- **Non-landslide sampling matters.** Importance depends on where the 0-labelled points sit. Test two random draws.
- **Random split optimism** applies here too. Consider the spatial split.
- **SHAP shows what the model uses, not what causes landslides.** Sanity-check each plot against mechanics, e.g. risk rising past about 25° of slope.
- This pushes the project away from soil mechanics. Keep it optional.

---

## Step 16: Write the report

**Time: about 3.5 h. Week 7.** Suggested outline:

1. **Introduction:** the landslide problem in Kerala, Wayanad 2024, objectives, and the susceptibility-vs-hazard scope statement.
2. **Study area:** geography, geology, rainfall, past events of the chosen district.
3. **Data:** inventory, factor layers, sources, resolution (30 m DEM, 10 m land cover, about 5 km rainfall), preprocessing and the master grid.
4. **Method:** FR definition, binning table, train/test split with seed, LSI, zoning method, ROC-AUC (success and prediction rate).
5. **Results:** FR tables and interpretation, susceptibility map, zone statistics, AUC, ROC figure.
6. **Discussion:** which factors matter and why (slope angle, weathering and lithology, rainfall, drainage), comparison with published AUC values (about 0.82-0.92), inventory bias.
7. **Limitations** (below) and **conclusions and future work:** hybrid AHP+ML, better inventory, rainfall-triggered hazard.
8. **Appendix:** FR tables, binning table, code (this repo).

**If your professor sees it as "just GIS" [Explainer].** The draft scored "stays geotechnical" 3/5. Emphasise geotechnical interpretation: slope angle, weathering and lithology, rainfall, drainage, and what each FR value implies about failure mechanisms.

---

## 28-hour plan

**[Explainer]** The draft gave a timeline only for the liquefaction project. This is my adaptation of the same 8-week, 3.5 h/week structure to 7B in Python. The draft estimated about 6 h of QGIS learning. Here that time goes to writing and debugging the scripts, which is faster if you are comfortable with numpy/pandas.

| Week | Hours | Tasks | Deliverable |
|---|---|---|---|
| 1 | 3.5 | Step 1 (district, proposal, approval), Step 2 (register, download inventory), Setup. | Proposal + inventory file |
| 2 | 3.5 | Step 3 (master grid), Step 4 (DEM → slope, aspect, curvature, with the aspect test). | 3 terrain rasters |
| 3 | 3.5 | Steps 5-7 (land cover, rainfall, lithology, distances, QC). | Aligned raster stack + VALID mask |
| 4 | 3.5 | Steps 8-10 (bins, split, non-landslide points, FR tables). | FR tables |
| 5 | 3.5 | Steps 11-12 (LSI, zones, map). | Susceptibility map |
| 6 | 3.5 | Steps 13-14 (ROC-AUC, zone validation, correlation/VIF, FR plots); optional Step 15. | ROC figure + checks |
| 7 | 3.5 | Step 16: write the report. | Draft report |
| 8 | 3.5 | Buffer for slippage, slides, viva preparation. | Final submission |

**Minimum viable version** (if time runs short): Weeks 1-5 plus Week 7, **5 factors**, no extras, about 21 hours. Add Step 13 at least in simple form (one prediction-rate AUC), because without validation the map is not a defensible result.

---

## Limitations, claims, risks, and verify list

### Limitations to state in the report

- **Independence assumption.** FR analyses each factor alone, then adds them.
- **Equal weights.** A weak factor (aspect) counts as much as a strong one (slope). The draft cites hybrid AHP+ML as suggested future work (PMC13534497).
- **Sensitivity to binning.** Different class boundaries change FR and the map.
- **Inventory quality.** Coarse, incomplete or roadside-biased inventories carry through to the map.
- **Random-split optimism.** Nearby landslides share conditions. A spatial split is stricter.
- **Static map.** It uses average rainfall, not a triggering storm, so it shows susceptibility, not hazard.
- **Resolution.** Rainfall at about 5 km and a 30 m DEM limit the detail.

### Claims you can and cannot make

| You can say | You should not say |
|---|---|
| The Kerala landslide inventory is coarse and no automated inventory exists for the state (PMC12749052, per the draft). | "First Wayanad study." ML studies of Wayanad already exist (HydroResearch 2025, per the draft). |
| Hybrid AHP+ML is suggested as future work in the literature (PMC13534497). | That your map is a hazard or risk map. It is susceptibility only. |
| Your AUC is comparable to or differs from published values of about 0.82-0.92, with reasons. | That a high AUC proves the map is correct everywhere. |

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Portal logins or slow downloads (OpenTopography, Bhukosh) | Register in Week 1. Use the bharatlas mirror for the inventory. Try the no-login DEM route (check it first). |
| Too few landslide points in the district | Choose a district with more points, or combine neighbouring districts. |
| Lithology data not found | Drop to 5 factors and state why. |
| Sign or alignment bugs in terrain layers | Run the synthetic-plane test (Step 4) and overlay points on rasters (Step 14). |
| Professor sees the project as "just GIS" | Emphasise geotechnical interpretation (Step 16). |
| Time slips | Follow the minimum viable version. Week 8 is buffer. |

### What to verify yourself

- The 12.6% landslide-prone figure is single-source **[V1]**. Check the NDMA NLRMP 2025 document.
- **Lithology data source.** The draft does not name one. Check GSI Bhukosh and district geology maps.
- AUC figures from PMC13534497 and PMC12847720. Read the papers, and copy their factor lists and binning.
- Bhukosh and bharatlas inventory formats and point counts for your chosen district.
- CHIRPS file naming, resolution and the best rainfall summary (mean annual vs monsoon) for a district-scale study.
- File names, tile names, column names and WorldCover class codes marked **[U: check]** above.
- Copernicus GLO-30 no-login bucket URL and tile naming, if you use that route.
- That your Geofabrik extract covers Kerala, and which layer/columns hold waterway and highway tags.

---

*Facts tagged [V1]/[V2]/[U] come from your research draft. Code, parameter choices (70/30 split, 300 m non-landslide buffer, bin edges, 5-class zoning) and the 7B timeline are my suggestions [Explainer]. Adjust them and record your choices in the report.*
