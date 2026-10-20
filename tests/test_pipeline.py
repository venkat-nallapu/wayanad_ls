"""Unit and integration test suite for Project 7B Phase 2:
Master grid reprojection, terrain derivatives, and OSM stream data.
"""
import unittest
import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from rasterio.transform import Affine
from scipy.ndimage import convolve

import common
from common import (
    RAW, PROC, OUT, MASTER_DIR, FACTORS_CONT_DIR, FACTORS_BIN_DIR, SAMPLES_DIR, FINAL_DIR, TABLES_DIR, FIGURES_DIR,
    CRS, RES, load_grid, load_tif, save_tif, get_master_grid_params
)
import importlib
from shapely.geometry import LineString, Point
from rasterio.transform import from_origin

terrain_mod = importlib.import_module("scripts.02_terrain_derivatives")
run_synthetic_plane_test = terrain_mod.run_synthetic_plane_test
prox_mod = importlib.import_module("scripts.03_proximity_rasters")
compute_proximity = prox_mod.compute_proximity
s11_mod = importlib.import_module("scripts.11_factor_correlation_vif")
s12_mod = importlib.import_module("scripts.12_spatial_split_validation")


class TestMasterGridAndDerivatives(unittest.TestCase):

    def test_01_osm_streams_exist_and_valid(self):
        """Verify that wayanad_streams.geojson was acquired and has valid geometries."""
        streams_path = RAW / "osm" / "wayanad_streams.geojson"
        self.assertTrue(streams_path.exists(), "wayanad_streams.geojson does not exist")
        
        gdf = gpd.read_file(streams_path)
        self.assertGreater(len(gdf), 1000, f"Expected >1000 streams, found {len(gdf)}")
        self.assertEqual(str(gdf.crs), "EPSG:4326")
        
        # Verify geometries are LineStrings or MultiLineStrings
        geom_types = set(gdf.geom_type.unique())
        self.assertTrue(geom_types.issubset({"LineString", "MultiLineString"}), f"Unexpected geom types: {geom_types}")
        
        # Check boundary intersection
        boundary_gdf = gpd.read_file(RAW / "boundary" / "wayanad_boundary.geojson")
        b_minx, b_miny, b_maxx, b_maxy = boundary_gdf.total_bounds
        s_minx, s_miny, s_maxx, s_maxy = gdf.total_bounds
        self.assertGreaterEqual(s_minx, b_minx - 0.05)
        self.assertLessEqual(s_maxx, b_maxx + 0.05)
        self.assertGreaterEqual(s_miny, b_miny - 0.05)
        self.assertLessEqual(s_maxy, b_maxy + 0.05)

    def test_02_common_grid_calculation(self):
        """Verify get_master_grid_params calculates snapped 30m grid bounds."""
        boundary_gdf = gpd.read_file(RAW / "boundary" / "wayanad_boundary.geojson")
        params = get_master_grid_params(boundary_gdf, res=30.0)
        
        self.assertEqual(params["crs"], "EPSG:32643")
        self.assertEqual(params["res"], 30.0)
        self.assertEqual(params["width"], 2438)
        self.assertEqual(params["height"], 1945)
        
        minx, miny, maxx, maxy = params["bounds"]
        self.assertEqual(minx % 30.0, 0.0)
        self.assertEqual(miny % 30.0, 0.0)
        self.assertEqual(maxx % 30.0, 0.0)
        self.assertEqual(maxy % 30.0, 0.0)

    def test_03_master_grid_files(self):
        """Verify grid.json, mask.tif, and dem_30m.tif in data/processed/master/."""
        grid_json_path = MASTER_DIR / "grid.json"
        mask_path = MASTER_DIR / "mask.tif"
        dem_path = MASTER_DIR / "dem_30m.tif"
        
        self.assertTrue(grid_json_path.exists())
        self.assertTrue(mask_path.exists())
        self.assertTrue(dem_path.exists())
        
        T, H, W = load_grid()
        self.assertEqual(H, 1945)
        self.assertEqual(W, 2438)
        self.assertIsInstance(T, Affine)
        
        mask = load_tif(mask_path)
        self.assertEqual(mask.shape, (1945, 2438))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertEqual(unique_vals, {0, 1})
        self.assertEqual(int(mask.sum()), 2365030)
        
        dem = load_tif(dem_path)
        self.assertEqual(dem.shape, (1945, 2438))
        self.assertEqual(dem.dtype, np.float32)
        
        # Verify elevation within mask
        district_elev = dem[mask == 1]
        self.assertEqual(int(np.isnan(district_elev).sum()), 0)
        self.assertGreater(float(np.min(district_elev)), 100.0)
        self.assertLess(float(np.max(district_elev)), 2300.0)

    def test_04_synthetic_plane_orientations(self):
        """Run synthetic plane aspect test across 8 compass directions and flat plane."""
        # This will raise AssertionError if any orientation or compass bin deviates
        run_synthetic_plane_test()

    def test_05_terrain_derivatives_outputs(self):
        """Verify continuous slope, aspect, and curvature factor GeoTIFFs."""
        slope_path = FACTORS_CONT_DIR / "slope.tif"
        aspect_path = FACTORS_CONT_DIR / "aspect.tif"
        curv_path = FACTORS_CONT_DIR / "curv.tif"
        
        self.assertTrue(slope_path.exists())
        self.assertTrue(aspect_path.exists())
        self.assertTrue(curv_path.exists())
        
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        
        # 1. Slope
        slope = load_tif(slope_path)
        self.assertEqual(slope.shape, (1945, 2438))
        slope_dist = slope[mask]
        self.assertEqual(int(np.isnan(slope_dist).sum()), 0)
        self.assertGreaterEqual(float(np.min(slope_dist)), 0.0)
        self.assertLess(float(np.max(slope_dist)), 70.0)
        self.assertAlmostEqual(float(np.mean(slope_dist)), 10.90, places=1)
        
        # 2. Aspect (continuous)
        aspect = load_tif(aspect_path)
        self.assertEqual(aspect.shape, (1945, 2438))
        aspect_dist = aspect[mask]
        self.assertEqual(int(np.isnan(aspect_dist).sum()), 0)
        self.assertEqual(float(np.min(aspect_dist)), -1.0)
        self.assertLessEqual(float(np.max(aspect_dist)), 360.0)
        
        # 3. Curvature
        curv = load_tif(curv_path)
        self.assertEqual(curv.shape, (1945, 2438))
        curv_dist = curv[mask]
        self.assertEqual(int(np.isnan(curv_dist).sum()), 0)
        self.assertLess(float(np.min(curv_dist)), -0.05)
        self.assertGreater(float(np.max(curv_dist)), 0.05)
        self.assertAlmostEqual(float(np.mean(curv_dist)), 0.0, places=4)

    def test_06_aspect_compass_binning(self):
        """Verify binned aspect GeoTIFFs (classes 1-9) in factors_binned and processed dirs."""
        from common import FACTORS_BIN_DIR
        bin_path = FACTORS_BIN_DIR / "aspect.tif"
        cls_path = PROC / "cls_aspect.tif"
        
        self.assertTrue(bin_path.exists(), "factors_binned/aspect.tif does not exist")
        self.assertTrue(cls_path.exists(), "data/processed/cls_aspect.tif does not exist")
        
        asp_bin = load_tif(bin_path)
        self.assertEqual(asp_bin.shape, (1945, 2438))
        self.assertEqual(asp_bin.dtype, np.uint8)
        
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        asp_bin_dist = asp_bin[mask]
        
        # Inside district, all cells must have valid classes 1..9
        unique_bins = set(np.unique(asp_bin_dist))
        self.assertEqual(unique_bins, set(range(1, 10)), f"Expected classes 1-9, found: {unique_bins}")
        
        # Flat count must match slope < 1.0 deg
        slope = load_tif(FACTORS_CONT_DIR / "slope.tif")[mask]
        flat_slope_count = int((slope < 1.0).sum())
        flat_bin_count = int((asp_bin_dist == 9).sum())
        self.assertEqual(flat_bin_count, flat_slope_count)

    def test_07_common_helpers_path_safety(self):
        """Verify save_tif and load_tif handle relative project paths without double-nesting."""
        import tempfile
        T, H, W = load_grid()
        dummy_arr = np.ones((5, 5), dtype=np.uint8)
        dummy_T = Affine(30.0, 0, 584220.0, 0, -30.0, 1324470.0)
        
        # 1. Test relative path with data/processed/
        saved = save_tif("data/processed/test_relative.tif", dummy_arr, dummy_T, nodata=0)
        self.assertTrue(saved.exists())
        self.assertNotIn("data/processed/data/processed", str(saved))
        
        # 2. Test load_tif with bare name and relative path
        loaded = load_tif("test_relative")
        self.assertEqual(loaded.shape, (5, 5))
        self.assertEqual(loaded[0, 0], 1)
        
        # Clean up test file
        if saved.exists():
            saved.unlink()

    def test_08_osm_roads_exist_and_valid(self):
        """Verify that wayanad_roads.geojson was acquired and has valid features."""
        roads_path = RAW / "osm" / "wayanad_roads.geojson"
        self.assertTrue(roads_path.exists(), "wayanad_roads.geojson does not exist")

        gdf = gpd.read_file(roads_path)
        self.assertGreater(len(gdf), 10000, f"Expected >10000 roads, found {len(gdf)}")
        self.assertEqual(str(gdf.crs), "EPSG:4326")

        # Verify linear geometries dominate
        line_count = int(gdf.geom_type.isin(["LineString", "MultiLineString"]).sum())
        self.assertGreater(line_count, 10200, f"Expected >10200 linear geometries, found {line_count}")

        # Check boundary bounds
        boundary_gdf = gpd.read_file(RAW / "boundary" / "wayanad_boundary.geojson")
        b_minx, b_miny, b_maxx, b_maxy = boundary_gdf.total_bounds
        r_minx, r_miny, r_maxx, r_maxy = gdf.total_bounds
        self.assertGreaterEqual(r_minx, b_minx - 0.05)
        self.assertLessEqual(r_maxx, b_maxx + 0.05)
        self.assertGreaterEqual(r_miny, b_miny - 0.05)
        self.assertLessEqual(r_maxy, b_maxy + 0.05)

    def test_09_proximity_rasters_output_and_integrity(self):
        """Verify continuous Euclidean distance rasters for roads and streams."""
        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)

        for name in ["dist_roads", "dist_streams", "d_road", "d_stream"]:
            path = FACTORS_CONT_DIR / f"{name}.tif"
            self.assertTrue(path.exists(), f"{path} does not exist")

        # 1. Dist to Roads
        dist_roads = load_tif(FACTORS_CONT_DIR / "dist_roads.tif")
        self.assertEqual(dist_roads.shape, (H, W))
        self.assertEqual(dist_roads.dtype, np.float32)

        r_valid = dist_roads[mask]
        self.assertEqual(int(np.isnan(r_valid).sum()), 0, "NaNs found in district for dist_roads")
        self.assertTrue(np.isnan(dist_roads[~mask]).all(), "Non-NaNs found outside district for dist_roads")
        self.assertEqual(float(np.nanmin(r_valid)), 0.0)
        self.assertGreater(float(np.nanmax(r_valid)), 6000.0)
        self.assertLess(float(np.nanmax(r_valid)), 10000.0)
        self.assertAlmostEqual(float(np.nanmean(r_valid)), 657.18, places=1)

        # 2. Dist to Streams
        dist_streams = load_tif(FACTORS_CONT_DIR / "dist_streams.tif")
        self.assertEqual(dist_streams.shape, (H, W))
        self.assertEqual(dist_streams.dtype, np.float32)

        s_valid = dist_streams[mask]
        self.assertEqual(int(np.isnan(s_valid).sum()), 0, "NaNs found in district for dist_streams")
        self.assertTrue(np.isnan(dist_streams[~mask]).all(), "Non-NaNs found outside district for dist_streams")
        self.assertEqual(float(np.nanmin(s_valid)), 0.0)
        self.assertGreater(float(np.nanmax(s_valid)), 3000.0)
        self.assertLess(float(np.nanmax(s_valid)), 5000.0)
        self.assertAlmostEqual(float(np.nanmean(s_valid)), 360.63, places=1)

    def test_10_landcover_raster_output_and_integrity(self):
        """Verify categorical ESA WorldCover 30m raster on master grid."""
        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)

        for path in [
            FACTORS_BIN_DIR / "landcover.tif",
            FACTORS_CONT_DIR / "landcover.tif",
            PROC / "landcover.tif",
        ]:
            self.assertTrue(path.exists(), f"{path} does not exist")

        lc = load_tif(FACTORS_BIN_DIR / "landcover.tif")
        self.assertEqual(lc.shape, (H, W))
        self.assertEqual(lc.dtype, np.uint8)

        # District mask checks
        lc_dist = lc[mask]
        self.assertEqual(int((lc_dist == 0).sum()), 0, "Unassigned (0) cells inside district for landcover")
        self.assertTrue((lc[~mask] == 0).all(), "Non-zero cells outside district for landcover")

        # Class breakdown checks
        unique_classes = set(np.unique(lc_dist))
        self.assertTrue(
            unique_classes.issubset({10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}) or
            unique_classes.issubset(set(range(1, 9))),
            f"Unexpected classes in landcover: {unique_classes}"
        )
        # Tree cover (code 10 in ESA, class 1 in contiguous) dominance in Western Ghats
        tree_pct = float(((lc_dist == 10) | (lc_dist == 1)).sum()) / len(lc_dist) * 100.0
        self.assertGreater(tree_pct, 85.0, f"Expected tree cover >85%, got {tree_pct:.2f}%")

    def test_11_all_layers_grid_alignment(self):
        """Verify all processed factor rasters share identical dimensions and mask consistency."""
        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        n_dist = int(mask.sum())
        self.assertEqual(n_dist, 2365030)

        factor_names = [
            ("master/dem_30m.tif", float, True),
            ("factors_continuous/slope.tif", float, True),
            ("factors_continuous/aspect.tif", float, True),
            ("factors_continuous/curvature.tif", float, True),
            ("factors_continuous/dist_roads.tif", float, True),
            ("factors_continuous/dist_streams.tif", float, True),
            ("factors_binned/aspect_binned.tif", int, False),
            ("factors_binned/landcover.tif", int, False),
        ]

        for rel_path, kind, is_float in factor_names:
            arr = load_tif(rel_path)
            self.assertEqual(arr.shape, (H, W), f"{rel_path} shape mismatch")
            if is_float:
                valid_count = int(np.isfinite(arr[mask]).sum())
            else:
                valid_count = int((arr[mask] > 0).sum())
            self.assertEqual(
                valid_count, n_dist,
                f"{rel_path} has missing values in district: {valid_count} vs {n_dist}"
            )

    def test_12_synthetic_proximity_edt(self):
        """Mathematically verify Euclidean distance transform, padding, and geometry filtering."""
        # 100 x 100 test grid at 30m resolution, origin (0, 3000)
        T_synth = from_origin(0.0, 3000.0, 30.0, 30.0)
        H_synth, W_synth = 100, 100

        # Case A: Horizontal line through row 50 center (y = 1485m)
        line_a = LineString([(-1000.0, 1485.0), (4000.0, 1485.0)])
        gdf_a = gpd.GeoDataFrame({"geometry": [line_a]}, crs="EPSG:32643")
        cropped_a, burned_a = compute_proximity(gdf_a, T_synth, H_synth, W_synth, pad=10, res=30.0)

        # Row 50 should be on the line (0m)
        self.assertEqual(float(cropped_a[50, :].min()), 0.0)
        self.assertEqual(float(cropped_a[50, :].max()), 0.0)
        # Rows 49 and 51 should be 1 pixel (30m) away
        self.assertEqual(float(cropped_a[49, :].min()), 30.0)
        self.assertEqual(float(cropped_a[51, :].min()), 30.0)
        # Rows 48 and 52 should be 2 pixels (60m) away
        self.assertEqual(float(cropped_a[48, :].min()), 60.0)
        self.assertEqual(float(cropped_a[52, :].min()), 60.0)

        # Case B: Line in the pad region outside the crop bounds (y = 3135m, 5 pixels North of row 0)
        line_b = LineString([(-1000.0, 3135.0), (4000.0, 3135.0)])
        gdf_b = gpd.GeoDataFrame({"geometry": [line_b]}, crs="EPSG:32643")
        cropped_b, _ = compute_proximity(gdf_b, T_synth, H_synth, W_synth, pad=10, res=30.0)

        # Row 0 of crop should see distance 5 * 30m = 150m
        self.assertEqual(float(cropped_b[0, :].min()), 150.0)
        self.assertEqual(float(cropped_b[1, :].min()), 180.0)

        # Case C: Ensure Point geometries are ignored
        point_c = Point(500.0, 1500.0)
        gdf_mixed = gpd.GeoDataFrame({"geometry": [line_a, point_c]}, crs="EPSG:32643")
        cropped_mixed, burned_mixed = compute_proximity(gdf_mixed, T_synth, H_synth, W_synth, pad=10, res=30.0)
        self.assertEqual(burned_mixed, burned_a)
        np.testing.assert_array_equal(cropped_mixed, cropped_a)

        # Case D: Empty or point-only GeoDataFrame raises ValueError
        gdf_point_only = gpd.GeoDataFrame({"geometry": [point_c]}, crs="EPSG:32643")
        with self.assertRaises(ValueError):
            compute_proximity(gdf_point_only, T_synth, H_synth, W_synth, pad=10, res=30.0)

    def test_13_landslide_inventory_spatial_overlay(self):
        """Verify landslide inventory samples against road, stream, and landcover rasters."""
        import rasterio

        inv_path = RAW / "inventory" / "wayanad_landslides.geojson"
        self.assertTrue(inv_path.exists(), "Landslide inventory GeoJSON does not exist")
        inv = gpd.read_file(inv_path).to_crs(CRS)

        # Intersect with district boundary mask
        mask = load_tif(MASTER_DIR / "mask.tif")
        with rasterio.open(MASTER_DIR / "mask.tif") as msrc:
            m_vals = np.array([val[0] for val in msrc.sample([(p.x, p.y) for p in inv.geometry])])

        district_inv = inv[m_vals == 1]
        self.assertEqual(len(district_inv), 204, f"Expected 204 landslides in district, got {len(district_inv)}")

        coords = [(p.x, p.y) for p in district_inv.geometry]

        # 1. Proximity to Roads
        with rasterio.open(FACTORS_CONT_DIR / "dist_roads.tif") as rsrc:
            r_vals = np.array([val[0] for val in rsrc.sample(coords)])
        self.assertEqual(int(np.isnan(r_vals).sum()), 0, "Found NaNs in road distance at landslide locations")
        self.assertEqual(float(np.nanmin(r_vals)), 0.0)
        # Landslides in mountainous terrain heavily cluster near roads (mean ~128m)
        self.assertLess(float(np.nanmean(r_vals)), 200.0)

        # 2. Proximity to Streams
        with rasterio.open(FACTORS_CONT_DIR / "dist_streams.tif") as ssrc:
            s_vals = np.array([val[0] for val in ssrc.sample(coords)])
        self.assertEqual(int(np.isnan(s_vals).sum()), 0, "Found NaNs in stream distance at landslide locations")
        self.assertEqual(float(np.nanmin(s_vals)), 0.0)
        self.assertLess(float(np.nanmean(s_vals)), 400.0)

        # 3. Land Cover
        with rasterio.open(FACTORS_BIN_DIR / "landcover.tif") as lcsrc:
            lc_vals = np.array([val[0] for val in lcsrc.sample(coords)])
        self.assertEqual(int((lc_vals == 0).sum()), 0, "Found unassigned land cover at landslide locations")
        # Tree cover (code 10 in ESA, class 1 in contiguous) should dominate the inventory
        tree_count = int(((lc_vals == 10) | (lc_vals == 1)).sum())
        self.assertGreater(tree_count / len(lc_vals), 0.80)

    def test_14_binned_factors_integrity(self):
        """Verify discretization and integer class integrity for all conditioning factors (Step 8)."""
        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        n_dist = int(mask.sum())
        self.assertEqual(n_dist, 2365030)

        # 1. Slope (5 bins)
        slope_bin = load_tif(FACTORS_BIN_DIR / "slope.tif")
        self.assertEqual(slope_bin.shape, (H, W))
        self.assertEqual(slope_bin.dtype, np.uint8)
        self.assertTrue((slope_bin[~mask] == 0).all())
        self.assertEqual(set(np.unique(slope_bin[mask])), {1, 2, 3, 4, 5})

        # 2. Curvature (3 bins)
        curv_bin = load_tif(FACTORS_BIN_DIR / "curvature.tif")
        self.assertEqual(curv_bin.shape, (H, W))
        self.assertEqual(curv_bin.dtype, np.uint8)
        self.assertTrue((curv_bin[~mask] == 0).all())
        self.assertEqual(set(np.unique(curv_bin[mask])), {1, 2, 3})

        # 3. Distance to Roads (5 bins)
        d_road_bin = load_tif(FACTORS_BIN_DIR / "dist_roads.tif")
        self.assertEqual(d_road_bin.shape, (H, W))
        self.assertEqual(d_road_bin.dtype, np.uint8)
        self.assertTrue((d_road_bin[~mask] == 0).all())
        self.assertEqual(set(np.unique(d_road_bin[mask])), {1, 2, 3, 4, 5})

        # 4. Distance to Streams (5 bins)
        d_stream_bin = load_tif(FACTORS_BIN_DIR / "dist_streams.tif")
        self.assertEqual(d_stream_bin.shape, (H, W))
        self.assertEqual(d_stream_bin.dtype, np.uint8)
        self.assertTrue((d_stream_bin[~mask] == 0).all())
        self.assertEqual(set(np.unique(d_stream_bin[mask])), {1, 2, 3, 4, 5})

        # 5. Aspect (9 bins)
        aspect_bin = load_tif(FACTORS_BIN_DIR / "aspect.tif")
        self.assertEqual(aspect_bin.shape, (H, W))
        self.assertEqual(aspect_bin.dtype, np.uint8)
        self.assertTrue((aspect_bin[~mask] == 0).all())
        self.assertEqual(set(np.unique(aspect_bin[mask])), set(range(1, 10)))

        # 6. Landcover contiguous (1..K) and mapping legend
        lc_bin = load_tif(FACTORS_BIN_DIR / "landcover.tif")
        self.assertEqual(lc_bin.shape, (H, W))
        self.assertEqual(lc_bin.dtype, np.uint8)
        self.assertTrue((lc_bin[~mask] == 0).all())
        self.assertEqual(int((lc_bin[mask] == 0).sum()), 0)
        self.assertTrue(set(np.unique(lc_bin[mask])).issubset(set(range(1, 9))))

        legend_path = FACTORS_BIN_DIR / "landcover_legend.json"
        self.assertTrue(legend_path.exists())
        legend_data = json.loads(legend_path.read_text())
        self.assertIn("esa_to_contiguous", legend_data)
        self.assertIn("contiguous_to_name", legend_data)
        self.assertEqual(len(legend_data["esa_to_contiguous"]), 8)

        # 7. Check bin edges definition file
        bin_edges_path = TABLES_DIR / "bin_edges.csv"
        self.assertTrue(bin_edges_path.exists())
        import pandas as pd
        bdf = pd.read_csv(bin_edges_path)
        self.assertIn("factor", bdf.columns)
        self.assertIn("class_id", bdf.columns)
        self.assertIn("class_label", bdf.columns)

    def test_15_dataset_split_and_negative_sampling(self):
        """Verify 70/30 split, 1:1 negative sampling, and >=300m exclusion constraint (Step 9)."""
        from scipy.spatial.distance import cdist

        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)

        # 1. Load points NPZ
        npz_path = SAMPLES_DIR / "points.npz"
        self.assertTrue(npz_path.exists(), "samples/points.npz does not exist")
        pts = np.load(npz_path)

        train_pos = pts["train"]
        val_pos = pts["val"]
        train_neg = pts["neg_tr"]
        val_neg = pts["neg_val"]

        # Exact sample count verification
        self.assertEqual(len(train_pos), 143, f"Expected 143 train positives, got {len(train_pos)}")
        self.assertEqual(len(val_pos), 61, f"Expected 61 val positives, got {len(val_pos)}")
        self.assertEqual(len(train_neg), 143, f"Expected 143 train negatives, got {len(train_neg)}")
        self.assertEqual(len(val_neg), 61, f"Expected 61 val negatives, got {len(val_neg)}")

        # 2. Verify all points are inside district mask
        for name, arr in [("train_pos", train_pos), ("val_pos", val_pos),
                          ("train_neg", train_neg), ("val_neg", val_neg)]:
            in_mask_count = int(mask[arr[:, 0], arr[:, 1]].sum())
            self.assertEqual(in_mask_count, len(arr), f"{name} contains points outside district mask!")

        # 3. Verify negative points uniqueness
        all_neg = np.vstack([train_neg, val_neg])
        unique_neg = np.unique(all_neg, axis=0)
        self.assertEqual(len(unique_neg), 204, "Duplicate negative samples generated!")

        # 4. Verify minimum exclusion distance >= 300m from ANY of the 204 landslides
        train_pos_xy = pts["train_xy"]
        val_pos_xy = pts["val_xy"]
        all_pos_xy = np.vstack([train_pos_xy, val_pos_xy])

        train_neg_xy = pts["neg_tr_xy"]
        val_neg_xy = pts["neg_val_xy"]
        all_neg_xy = np.vstack([train_neg_xy, val_neg_xy])

        pairwise_dist = cdist(all_neg_xy, all_pos_xy)
        min_observed_dist = float(np.min(pairwise_dist))
        self.assertGreaterEqual(
            min_observed_dist, 300.0,
            f"Exclusion constraint violated! Minimum distance observed: {min_observed_dist:.2f} m < 300.0 m"
        )

        # 5. Verify sample GeoJSON and CSV exports
        for fname in ["samples_all.geojson", "samples_all.csv", "train_samples.geojson",
                      "val_samples.geojson", "train_landslides.geojson", "train_negatives.geojson"]:
            p = SAMPLES_DIR / fname
            self.assertTrue(p.exists(), f"Sample export {p} does not exist")

        all_gdf = gpd.read_file(SAMPLES_DIR / "samples_all.geojson")
        self.assertEqual(len(all_gdf), 408)
        self.assertEqual(set(all_gdf.split.unique()), {"train", "val"})
        self.assertEqual(set(all_gdf.label.unique()), {0, 1})

    def test_16_frequency_ratio_computation(self):
        """Verify vectorized FR formulas, area/slide conservation, and table exports (Step 10)."""
        import pandas as pd

        # Check export files exist
        csv_path = TABLES_DIR / "fr_summary.csv"
        md_path = TABLES_DIR / "fr_summary.md"
        tex_path = TABLES_DIR / "fr_summary.tex"

        self.assertTrue(csv_path.exists(), "reports/tables/fr_summary.csv does not exist")
        self.assertTrue(md_path.exists(), "reports/tables/fr_summary.md does not exist")
        self.assertTrue(tex_path.exists(), "reports/tables/fr_summary.tex does not exist")

        df = pd.read_csv(csv_path)
        required_cols = {"factor", "class_id", "bin", "class_label", "pixels", "slides", "area_pct", "slide_pct", "FR"}
        self.assertTrue(required_cols.issubset(set(df.columns)), f"Missing columns in FR summary: {df.columns}")
        self.assertTrue((df["bin"] == df["class_id"]).all(), "bin and class_id values do not match")

        expected_factors = {"slope", "aspect", "curvature", "dist_roads", "dist_streams", "landcover"}
        self.assertEqual(set(df.factor.unique()), expected_factors)

        for factor, grp in df.groupby("factor"):
            # Area % must sum to 100%
            self.assertAlmostEqual(float(grp.area_pct.sum()), 100.0, places=4,
                                   msg=f"Area % does not sum to 100 for {factor}")
            # Slide % must sum to 100%
            self.assertAlmostEqual(float(grp.slide_pct.sum()), 100.0, places=4,
                                   msg=f"Slide % does not sum to 100 for {factor}")

            # Total district pixels must be 2,365,030
            self.assertEqual(int(grp.pixels.sum()), 2365030,
                             msg=f"Pixel total mismatch for {factor}")
            # Total training slides must be 143
            self.assertEqual(int(grp.slides.sum()), 143,
                             msg=f"Training slide count mismatch for {factor}")

            # FR formula identity: FR = slide_pct / area_pct
            for _, row in grp.iterrows():
                expected_fr = row["slide_pct"] / row["area_pct"]
                self.assertAlmostEqual(float(row["FR"]), float(expected_fr), places=5,
                                       msg=f"FR formula mismatch for {factor} class {row['class_id']}")

        # Ensure Markdown and LaTeX content are non-empty
        self.assertGreater(len(md_path.read_text()), 500)
        self.assertGreater(len(tex_path.read_text()), 500)

    def test_17_lsi_raster_properties(self):
        """Verify LSI raster properties, finite masking, and theoretical mean (Step 11)."""
        lsi_paths = [FINAL_DIR / "lsi.tif", PROC / "lsi.tif"]
        for p in lsi_paths:
            self.assertTrue(p.exists(), f"LSI raster does not exist at {p}")

        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        lsi = load_tif(FINAL_DIR / "lsi.tif")

        self.assertEqual(lsi.shape, (H, W), "LSI shape does not match master grid")

        # Finite inside mask and NaN outside
        valid_cells = lsi[mask]
        invalid_cells = lsi[~mask]

        self.assertEqual(len(valid_cells), 2365030, "Valid mask cell count mismatch")
        self.assertTrue(np.isfinite(valid_cells).all(), "LSI contains non-finite values inside district boundary")
        self.assertTrue(np.isnan(invalid_cells).all(), "LSI contains non-NaN values outside district boundary")

        # Range and theoretical expectation
        lsi_min = float(valid_cells.min())
        lsi_max = float(valid_cells.max())
        lsi_mean = float(valid_cells.mean())

        self.assertGreater(lsi_min, 0.0, "LSI minimum should be positive")
        self.assertLess(lsi_max, 30.0, "LSI maximum exceeds realistic bound")
        # In FR, area-weighted mean of each factor is 1.0, so sum of 6 factors has expected mean 6.0
        self.assertAlmostEqual(lsi_mean, 6.0, delta=0.05,
                               msg=f"LSI mean {lsi_mean:.4f} diverges from theoretical expectation of 6.000")

    def test_18_susceptibility_zones_classification(self):
        """Verify 5-class Fisher-Jenks classification, masking, guide compatibility, and map exports (Step 12)."""
        zone_paths = [
            FINAL_DIR / "susceptibility_zones_jenks.tif",
            PROC / "susceptibility_zones.tif",
            PROC / "zones.tif",
            FINAL_DIR / "zones.tif",
        ]
        for p in zone_paths:
            self.assertTrue(p.exists(), f"Susceptibility zone raster does not exist at {p}")

        # Verify guide alias load_tif('zones') works seamlessly
        z_guide = load_tif("zones")
        z_alias = load_tif("susceptibility_zones")
        np.testing.assert_array_equal(z_guide, z_alias, "zones.tif and susceptibility_zones.tif differ")

        T, H, W = load_grid()
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        zone = load_tif(FINAL_DIR / "susceptibility_zones_jenks.tif")

        self.assertEqual(zone.shape, (H, W), "Zone raster shape mismatch")

        # Inside mask strictly in 1..5; outside mask strictly 0
        valid_zones = zone[mask]
        invalid_zones = zone[~mask]

        self.assertEqual(set(np.unique(valid_zones)), {1, 2, 3, 4, 5},
                         "Susceptibility zones inside mask must strictly contain classes 1, 2, 3, 4, 5")
        self.assertTrue((invalid_zones == 0).all(), "Outside mask must be strictly 0 (NoData)")

        # Verify publication map figure generation
        for fname in ["susceptibility_map.png", "susceptibility_zones_jenks.png"]:
            fig_path = FIGURES_DIR / fname
            self.assertTrue(fig_path.exists(), f"Publication figure {fig_path} does not exist")
            self.assertGreater(fig_path.stat().st_size, 50000, f"Figure {fig_path} file size is suspiciously small")

    def test_19_monotonic_landslide_density(self):
        """Verify strict monotonic increasing landslide density across susceptibility zones (Step 12)."""
        import pandas as pd

        stats_path = TABLES_DIR / "zone_statistics.csv"
        self.assertTrue(stats_path.exists(), "zone_statistics.csv does not exist")

        df = pd.read_csv(stats_path)
        required_cols = {
            "zone_id", "zone", "lsi_range", "cell_count", "area_km2", "area_pct",
            "train_slides", "val_slides", "total_slides", "total_density_per_km2",
            "train_density_per_km2", "val_density_per_km2", "density_ratio"
        }
        self.assertTrue(required_cols.issubset(set(df.columns)), f"Missing columns in zone statistics: {df.columns}")

        # Area conservation
        self.assertEqual(int(df["cell_count"].sum()), 2365030, "Total classified cells mismatch")
        self.assertAlmostEqual(float(df["area_pct"].sum()), 100.0, places=1)

        # Slide counts
        self.assertEqual(int(df["train_slides"].sum()), 143, "Training slide sum mismatch")
        self.assertEqual(int(df["val_slides"].sum()), 61, "Validation slide sum mismatch")
        self.assertEqual(int(df["total_slides"].sum()), 204, "Total slide sum mismatch")

        # STRICT MONOTONIC DENSITY CHECK:
        # Density(Very High) > Density(High) > Density(Moderate) > Density(Low) > Density(Very Low)
        total_densities = df["total_density_per_km2"].values
        train_densities = df["train_density_per_km2"].values
        val_densities = df["val_density_per_km2"].values

        for i in range(len(total_densities) - 1):
            self.assertLess(
                total_densities[i], total_densities[i + 1],
                f"Total landslide density non-monotonic at class {i+1} -> {i+2}: "
                f"{total_densities[i]:.4f} >= {total_densities[i+1]:.4f}"
            )
            self.assertLess(
                train_densities[i], train_densities[i + 1],
                f"Training landslide density non-monotonic at class {i+1} -> {i+2}: "
                f"{train_densities[i]:.4f} >= {train_densities[i+1]:.4f}"
            )
            # Validation density non-decreasing across all classes
            self.assertLessEqual(
                val_densities[i], val_densities[i + 1],
                f"Validation landslide density decreased at class {i+1} -> {i+2}: "
                f"{val_densities[i]:.4f} > {val_densities[i+1]:.4f}"
            )

        # Validation density strictly increases across hazard classes 2 -> 3 -> 4 -> 5
        for i in range(1, len(val_densities) - 1):
            self.assertLess(
                val_densities[i], val_densities[i + 1],
                f"Validation landslide density non-increasing at class {i+1} -> {i+2}: "
                f"{val_densities[i]:.4f} >= {val_densities[i+1]:.4f}"
            )

        # Verify zone_validation.csv exists and has monotonic density ratio
        zone_val_path = OUT / "zone_validation.csv"
        self.assertTrue(zone_val_path.exists(), "zone_validation.csv does not exist")
        zv_df = pd.read_csv(zone_val_path)
        d_ratios = zv_df["density_ratio"].values
        for i in range(1, len(d_ratios) - 1):
            self.assertLess(
                d_ratios[i], d_ratios[i + 1],
                f"Validation density ratio non-increasing at class {i+1} -> {i+2}"
            )

        # High + Very High zones must capture majority of landslides (>70%)
        high_risk_slides = df.iloc[3:]["total_slides"].sum()
        high_risk_pct = 100.0 * high_risk_slides / 204
        self.assertGreater(high_risk_pct, 70.0, f"High risk zones capture only {high_risk_pct:.1f}% of landslides")

    def test_20_roc_auc_and_validation_performance(self):
        """Verify ROC-AUC scores (>0.80), cumulative curves, and 2024 Chooralmala benchmark (Step 13)."""
        import pandas as pd
        from pyproj import Transformer
        from rasterio.transform import rowcol
        from sklearn.metrics import roc_auc_score

        # 1. Output figures and metrics tables existence
        self.assertTrue((FIGURES_DIR / "roc_curves.png").exists(), "roc_curves.png does not exist")
        self.assertTrue((FIGURES_DIR / "cumulative_susceptibility.png").exists(), "cumulative_susceptibility.png does not exist")
        self.assertTrue((TABLES_DIR / "validation_metrics.csv").exists(), "validation_metrics.csv does not exist")
        self.assertTrue((TABLES_DIR / "validation_metrics.md").exists(), "validation_metrics.md does not exist")

        # 2. Recompute and verify ROC-AUC scores directly
        lsi = load_tif(FINAL_DIR / "lsi.tif")
        pts = np.load(SAMPLES_DIR / "points.npz")

        train_pos = pts["train"]
        val_pos = pts["val"]
        train_neg = pts["neg_tr"]
        val_neg = pts["neg_val"]

        y_train = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
        s_train = np.r_[lsi[train_pos[:, 0], train_pos[:, 1]], lsi[train_neg[:, 0], train_neg[:, 1]]]
        auc_train = float(roc_auc_score(y_train, s_train))

        y_val = np.r_[np.ones(len(val_pos)), np.zeros(len(val_neg))]
        s_val = np.r_[lsi[val_pos[:, 0], val_pos[:, 1]], lsi[val_neg[:, 0], val_neg[:, 1]]]
        auc_val = float(roc_auc_score(y_val, s_val))

        # Scientific defensibility threshold: AUC > 0.80
        self.assertGreater(auc_train, 0.80, f"Training Success Rate AUC {auc_train:.4f} < 0.80")
        self.assertGreater(auc_val, 0.80, f"Validation Prediction Rate AUC {auc_val:.4f} < 0.80")
        self.assertGreaterEqual(auc_val, 0.85, f"Validation AUC {auc_val:.4f} below expected 0.85 benchmark")

        # 3. Verify Geomorphological Cumulative Susceptibility Curve Performance
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        valid_lsi = lsi[mask]
        area_fractions = np.linspace(0, 1, 1001)
        lsi_thresholds = np.quantile(valid_lsi, 1 - area_fractions)

        val_lsi_vals = lsi[val_pos[:, 0], val_pos[:, 1]]
        cum_val_capture = np.array([(val_lsi_vals >= th).mean() for th in lsi_thresholds])
        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_cum_val = float(trap_fn(cum_val_capture, area_fractions))

        self.assertGreater(auc_cum_val, 0.85, f"Cumulative prediction AUC {auc_cum_val:.4f} < 0.85")
        # Top 20% area should capture >= 75% unseen validation landslides
        val_top20_capture = cum_val_capture[200]
        self.assertGreaterEqual(val_top20_capture, 0.75,
                                f"Top 20% area captured only {val_top20_capture*100:.1f}% validation landslides")

        # 4. Verify validation_metrics.csv table values
        m_df = pd.read_csv(TABLES_DIR / "validation_metrics.csv")
        m_dict = dict(zip(m_df["metric"], m_df["value"]))
        self.assertIn("Success Rate ROC-AUC (Train)", m_dict)
        self.assertIn("Prediction Rate ROC-AUC (Val)", m_dict)
        self.assertIn("2024 Chooralmala Disaster LSI", m_dict)

        # 5. Chooralmala Real-World Out-of-Sample Catastrophic Disaster Benchmark
        # Coordinate: approx 11.543°N, 76.155°E (crown failure initiation of July 30, 2024)
        transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        ch_utm_x, ch_utm_y = transformer.transform(76.155, 11.543)

        T, H, W = load_grid()
        r, c = rowcol(T, ch_utm_x, ch_utm_y)
        self.assertTrue(0 <= r < H and 0 <= c < W, "Chooralmala pixel falls outside master grid")

        ch_lsi = float(lsi[r, c])
        percentile = float((valid_lsi < ch_lsi).mean() * 100.0)

        # Must be in top decile of district susceptibility (> 80th percentile)
        self.assertGreater(percentile, 80.0,
                           f"Chooralmala percentile {percentile:.1f}% is below 80th percentile")

        zone = load_tif(FINAL_DIR / "susceptibility_zones_jenks.tif")
        ch_zone = int(zone[r, c])
        # Must be classified in High (4) or Very High (5) susceptibility
        self.assertIn(ch_zone, (4, 5),
                      f"Chooralmala classified into unexpected zone {ch_zone} (expected 4 or 5)")

    def test_21_factor_correlation_and_vif(self):
        """Verify factor independence (|r| < 0.70) and low multicollinearity (VIF < 5.0) (Step 14)."""
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        # 1. Output files exist
        self.assertTrue((TABLES_DIR / "vif_multicollinearity.csv").exists(), "vif_multicollinearity.csv missing")
        self.assertTrue((TABLES_DIR / "vif_multicollinearity.md").exists(), "vif_multicollinearity.md missing")
        self.assertTrue((FIGURES_DIR / "correlation_heatmap.png").exists(), "correlation_heatmap.png missing")
        self.assertTrue((OUT / "correlation.csv").exists(), "outputs/correlation.csv missing")
        self.assertTrue((OUT / "vif.csv").exists(), "outputs/vif.csv missing")

        # 2. Verify stored VIF table values
        vif_df = pd.read_csv(TABLES_DIR / "vif_multicollinearity.csv")
        self.assertEqual(len(vif_df), 5, "Expected 5 continuous conditioning factors in VIF table")
        expected_factors = {"slope", "aspect", "curvature", "dist_roads", "dist_streams"}
        self.assertEqual(set(vif_df["factor"]), expected_factors)

        for _, row in vif_df.iterrows():
            vif_val = float(row["VIF"])
            self.assertLess(vif_val, 5.0, f"Factor {row['factor']} has VIF {vif_val:.3f} >= 5.0")
            self.assertLess(vif_val, 1.5, f"Factor {row['factor']} has unexpectedly high VIF {vif_val:.3f}")

        # 3. Verify stored correlation matrix
        corr_df = pd.read_csv(OUT / "correlation.csv", index_col=0)
        corr_vals = corr_df.values
        upper_tri = np.triu(np.abs(corr_vals), k=1)
        max_r = float(upper_tri.max())
        self.assertLess(max_r, 0.70, f"Max correlation {max_r:.4f} violates independence threshold of 0.70")
        self.assertAlmostEqual(max_r, 0.4194, places=2)

        # 4. Programmatic execution check: verify evaluate_multicollinearity runs on sample
        corr_sample, vif_sample_df = s11_mod.evaluate_multicollinearity(sample_size=50000, save_outputs=False)
        upper_tri_sample = np.triu(np.abs(corr_sample.values), k=1)
        self.assertLess(float(upper_tri_sample.max()), 0.70)
        for _, row in vif_sample_df.iterrows():
            self.assertLess(float(row["VIF"]), 5.0)

    def test_22_roadside_reporting_bias(self):
        """Verify quantification of roadside reporting and observation bias (Step 14)."""
        import pandas as pd

        # Deliverables exist
        self.assertTrue((TABLES_DIR / "roadside_bias_metrics.csv").exists(), "roadside_bias_metrics.csv missing")
        self.assertTrue((TABLES_DIR / "roadside_bias_metrics.md").exists(), "roadside_bias_metrics.md missing")
        self.assertTrue((FIGURES_DIR / "road_proximity_bias.png").exists(), "road_proximity_bias.png missing")
        self.assertTrue((OUT / "roadside_bias.csv").exists(), "outputs/roadside_bias.csv missing")

        # Run evaluate_roadside_bias() directly to confirm execution
        bdf = s11_mod.evaluate_roadside_bias()
        tier1 = bdf[bdf["tier_id"] == 1].iloc[0]

        # Immediate roadside corridor (<50m):
        # 58.74% of training slides vs 24.11% of district area -> FR = 2.437
        self.assertAlmostEqual(float(tier1["train_slide_pct"]), 58.74, places=1)
        self.assertAlmostEqual(float(tier1["area_pct"]), 24.11, places=1)
        self.assertGreater(float(tier1["fr_train"]), 2.0)

        # Total inventory slides in 0-50m buffer exceeds 60%
        self.assertGreater(float(tier1["total_slide_pct"]), 60.0)

        # Verify negative correlation at remote distances (>300m FR < 0.5)
        tier5 = bdf[bdf["tier_id"] == 5].iloc[0]
        self.assertLess(float(tier5["fr_train"]), 0.5)

        # Verify continuous distance statistics
        mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)
        dist_roads = load_tif("dist_roads")
        pts = np.load(SAMPLES_DIR / "points.npz")
        all_pos = np.vstack([pts["train"], pts["val"]])

        med_grid = float(np.median(dist_roads[mask]))
        med_slides = float(np.median(dist_roads[all_pos[:, 0], all_pos[:, 1]]))

        self.assertGreaterEqual(med_grid, 100.0, "District median road distance should be >= 100m")
        self.assertLessEqual(med_slides, 50.0, "Landslide median road distance should be <= 50m")
        self.assertGreater(med_grid / med_slides, 3.0, "District median distance must be >3x landslide median distance")

    def test_23_spatial_split_validation(self):
        """Verify spatial block cross-validation runs and produces defensible spatial AUCs (Step 15)."""
        import pandas as pd

        # Deliverables exist
        self.assertTrue((TABLES_DIR / "spatial_split_metrics.csv").exists(), "spatial_split_metrics.csv missing")
        self.assertTrue((TABLES_DIR / "spatial_split_metrics.md").exists(), "spatial_split_metrics.md missing")
        self.assertTrue((FIGURES_DIR / "spatial_split_map.png").exists(), "spatial_split_map.png missing")

        # RUN THE ACTUAL SPATIAL SPLIT CROSS-VALIDATION FUNCTION
        metrics_df = s12_mod.evaluate_spatial_split(save_outputs=False)
        self.assertIsNotNone(metrics_df)
        self.assertIn("results", metrics_df.attrs)
        res = metrics_df.attrs["results"]

        auc_fold1 = res["auc_n_to_s"]
        auc_fold2 = res["auc_s_to_n"]
        mean_spatial_auc = res["mean_spatial_auc"]
        random_auc = res["random_val_auc"]
        optimism = res["optimism_gap"]
        top20_s = res["top20_n_to_s"] * 100.0
        top20_n = res["top20_s_to_n"] * 100.0

        # Scientific defensibility: out-of-block spatial prediction AUCs must exceed 0.80
        self.assertGreater(auc_fold1, 0.80, f"Fold 1 spatial AUC {auc_fold1:.4f} < 0.80")
        self.assertGreater(auc_fold2, 0.80, f"Fold 2 spatial AUC {auc_fold2:.4f} < 0.80")
        self.assertGreater(mean_spatial_auc, 0.80, f"Mean spatial AUC {mean_spatial_auc:.4f} < 0.80")

        # Confirm reduction of spatial autocorrelation optimism:
        # Spatial block AUC must be strictly lower than random 70/30 split AUC
        self.assertLess(mean_spatial_auc, random_auc,
                        f"Spatial AUC {mean_spatial_auc:.4f} should be strictly lower than random split {random_auc:.4f}")
        self.assertGreater(optimism, 0.02,
                           f"Expected spatial autocorrelation optimism gap > 0.02, got {optimism:.4f}")

        # Top 20% area capture rate in spatial blocks exceeds 70%
        self.assertGreaterEqual(top20_s, 70.0)
        self.assertGreaterEqual(top20_n, 70.0)

    def test_24_quality_checks_and_sanity_tests(self):
        """Verify Step 14 class sizes, zero-FR identification, and point-raster alignment sanity checks."""
        import pandas as pd

        # 1. Point-raster coordinate alignment check
        aligned = s11_mod.verify_point_raster_alignment()
        self.assertTrue(aligned, "Point-raster alignment verification failed")

        # 2. Class sample size and zero-FR evaluation
        small_classes = s11_mod.evaluate_class_sample_sizes()
        self.assertIsNotNone(small_classes)

        # Verify that slope class 5 (>40 deg) is identified with 0 slides and FR = 0.0
        slope_zero = small_classes[(small_classes["factor"] == "slope") & (small_classes["bin"] == 5)]
        self.assertEqual(len(slope_zero), 1, "Slope class 5 (>40 deg) not found in small classes")
        self.assertEqual(int(slope_zero.iloc[0]["slides"]), 0)
        self.assertEqual(float(slope_zero.iloc[0]["FR"]), 0.0)

        # 3. Verify sanity check tables and figures exist
        self.assertTrue((TABLES_DIR / "small_class_sanity_check.csv").exists(), "small_class_sanity_check.csv missing")
        self.assertTrue((TABLES_DIR / "small_class_sanity_check.md").exists(), "small_class_sanity_check.md missing")
        self.assertTrue((FIGURES_DIR / "fr_by_factor.png").exists(), "fr_by_factor.png missing")
        self.assertGreater((FIGURES_DIR / "fr_by_factor.png").stat().st_size, 20000)


if __name__ == "__main__":
    unittest.main()


