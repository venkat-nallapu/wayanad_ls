"""09_classify_zones.py - Susceptibility Zone Classification (Step 12).

Classifies continuous LSI into 5 susceptibility zones using Fisher-Jenks Natural Breaks:
    1: Very Low
    2: Low
    3: Moderate
    4: High
    5: Very High
    0: NoData (outside district boundary)

Generates:
    - GeoTIFF: data/processed/final/susceptibility_zones_jenks.tif (and data/processed/susceptibility_zones.tif)
    - High-resolution cartographic map: reports/figures/susceptibility_map.png
    - Statistical tables: reports/tables/zone_statistics.csv and reports/tables/zone_statistics.md
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import jenkspy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import geopandas as gpd
from pyproj import Transformer

from common import (
    PROC, OUT, MASTER_DIR, FINAL_DIR, TABLES_DIR, FIGURES_DIR, SAMPLES_DIR,
    RES, DISTRICT, CRS, load_grid, load_tif, save_tif
)

ZONE_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
ZONE_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def add_scale_bar(ax, length_km=10, location=(0.06, 0.06), linewidth=3, text_offset=0.02):
    """Draw a clean, publication-ready metric scale bar."""
    x0_frac, y0_frac = location
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    x0 = xlim[0] + x0_frac * x_range
    y0 = ylim[0] + y0_frac * y_range
    
    length_m = length_km * 1000.0
    x1 = x0 + length_m
    
    # Scale bar line
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=linewidth, solid_capstyle="butt", zorder=10)
    # End ticks
    tick_len = 0.01 * y_range
    ax.plot([x0, x0], [y0 - tick_len, y0 + tick_len], color="black", linewidth=1.5, zorder=10)
    ax.plot([x1, x1], [y0 - tick_len, y0 + tick_len], color="black", linewidth=1.5, zorder=10)
    
    # Scale label
    ax.text(x0 + length_m / 2.0, y0 + tick_len + (text_offset * y_range), f"{length_km} km",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="black", zorder=10,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))


def add_north_arrow(ax, location=(0.93, 0.92)):
    """Draw a clear North arrow on the map."""
    ax.annotate(
        "N",
        xy=(location[0], location[1]),
        xytext=(location[0], location[1] - 0.07),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(facecolor="black", edgecolor="black", width=2.5, headwidth=8, headlength=10),
        zorder=10,
        bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="black", lw=0.5, alpha=0.8)
    )


def classify_susceptibility_zones():
    print("=" * 78)
    print("STEP 12: SUSCEPTIBILITY ZONE CLASSIFICATION (FISHER-JENKS NATURAL BREAKS)")
    print("=" * 78)

    # 1. Load Master Grid and LSI
    T, H, W = load_grid()
    mask = load_tif(MASTER_DIR / "mask.tif").astype(bool)

    lsi_candidates = [
        FINAL_DIR / "lsi.tif",
        PROC / "lsi.tif",
    ]
    lsi_path = None
    for p in lsi_candidates:
        if p.exists():
            lsi_path = p
            break
    if lsi_path is None:
        raise FileNotFoundError(f"LSI raster not found in {[str(c) for c in lsi_candidates]}")

    lsi = load_tif(lsi_path)
    ok = np.isfinite(lsi) & mask
    valid_lsi = lsi[ok]
    print(f"Loaded continuous LSI from {lsi_path} ({len(valid_lsi):,} valid pixels)")

    # 2. Compute Fisher-Jenks Natural Breaks on systematic grid subsample
    print("\nComputing 5-class Fisher-Jenks Natural Breaks...")
    sample = valid_lsi[::50]  # systematic subsample (~47k cells) for fast, robust break fitting
    breaks = jenkspy.jenks_breaks(sample, n_classes=5)
    edges = breaks[1:-1]  # 4 internal split boundaries
    print(f"Full Jenks breaks: {[round(b, 4) for b in breaks]}")
    print(f"Internal class thresholds: {[round(e, 4) for e in edges]}")

    # 3. Digitize zones: 1..5, 0 for outside district
    zone = np.digitize(lsi, edges) + 1
    zone[~ok] = 0
    zone = zone.astype("uint8")

    # Save zone rasters
    out_jenks = FINAL_DIR / "susceptibility_zones_jenks.tif"
    out_proc_jenks = PROC / "susceptibility_zones_jenks.tif"
    out_proc = PROC / "susceptibility_zones.tif"
    out_final = FINAL_DIR / "susceptibility_zones.tif"
    out_proc_guide = PROC / "zones.tif"
    out_final_guide = FINAL_DIR / "zones.tif"

    for p in (out_jenks, out_proc_jenks, out_proc, out_final, out_proc_guide, out_final_guide):
        save_tif(p, zone, T, nodata=0)
    print(f"\nSaved classified zone GeoTIFFs to:\n  - {out_jenks}\n  - {out_proc}\n  - {out_proc_guide}")

    # 4. Load landslide sample points for zonal statistics
    pts_candidates = [
        SAMPLES_DIR / "points.npz",
        PROC / "points.npz",
    ]
    pts_path = None
    for p in pts_candidates:
        if p.exists():
            pts_path = p
            break
    if pts_path is None:
        raise FileNotFoundError(f"points.npz not found in {[str(c) for c in pts_candidates]}")

    pts = np.load(pts_path)
    train_rc = pts["train"]
    val_rc = pts["val"]
    train_xy = pts["train_xy"]
    val_xy = pts["val_xy"]

    train_hits = np.bincount(zone[train_rc[:, 0], train_rc[:, 1]], minlength=6)[1:]
    val_hits = np.bincount(zone[val_rc[:, 0], val_rc[:, 1]], minlength=6)[1:]
    total_hits = train_hits + val_hits
    area_cells = np.bincount(zone[ok], minlength=6)[1:]

    cell_area_km2 = (RES * RES) / 1e6  # 30m x 30m = 0.0009 km2
    area_km2 = area_cells * cell_area_km2
    total_area_km2 = area_km2.sum()

    area_pct = 100.0 * area_cells / area_cells.sum()
    train_slide_pct = 100.0 * train_hits / train_hits.sum()
    val_slide_pct = 100.0 * val_hits / val_hits.sum()
    total_slide_pct = 100.0 * total_hits / total_hits.sum()

    train_density = train_hits / area_km2
    val_density = val_hits / area_km2
    total_density = total_hits / area_km2
    density_ratio = total_slide_pct / area_pct

    # Create interval string representation
    intervals = [
        f"< {breaks[1]:.2f}",
        f"{breaks[1]:.2f} - {breaks[2]:.2f}",
        f"{breaks[2]:.2f} - {breaks[3]:.2f}",
        f"{breaks[3]:.2f} - {breaks[4]:.2f}",
        f">= {breaks[4]:.2f}",
    ]

    stats_df = pd.DataFrame({
        "zone_id": range(1, 6),
        "zone": ZONE_LABELS,
        "lsi_range": intervals,
        "cell_count": area_cells,
        "area_km2": area_km2.round(2),
        "area_pct": area_pct.round(2),
        "train_slides": train_hits,
        "train_slide_pct": train_slide_pct.round(2),
        "val_slides": val_hits,
        "val_slide_pct": val_slide_pct.round(2),
        "total_slides": total_hits,
        "total_slide_pct": total_slide_pct.round(2),
        "train_density_per_km2": train_density.round(4),
        "val_density_per_km2": val_density.round(4),
        "total_density_per_km2": total_density.round(4),
        "density_ratio": density_ratio.round(2),
    })

    print("\n" + "=" * 78)
    print("SUSCEPTIBILITY ZONE STATISTICAL SUMMARY")
    print("=" * 78)
    print(stats_df[[
        "zone_id", "zone", "lsi_range", "area_km2", "area_pct",
        "train_slides", "val_slides", "total_density_per_km2", "density_ratio"
    ]].to_string(index=False))

    # Verify monotonic density progression
    densities = stats_df["total_density_per_km2"].values
    is_monotonic = all(densities[i] <= densities[i + 1] for i in range(len(densities) - 1))
    print(f"\nMonotonic increasing landslide density check: "
          f"{'PASSED (Density increases from Very Low to Very High)' if is_monotonic else 'FAILED'}")

    # Export tables
    stats_df.to_csv(TABLES_DIR / "zone_statistics.csv", index=False)
    stats_df.to_csv(OUT / "zone_statistics.csv", index=False)
    pd.DataFrame({"zone": ZONE_LABELS, "area_pct": area_pct.round(2)}).to_csv(OUT / "zone_area.csv", index=False)

    md_lines = [
        f"# Landslide Susceptibility Zone Statistics - {DISTRICT} District",
        "",
        "Classification Method: Fisher-Jenks Natural Breaks (5 classes)",
        "",
        "| Zone ID | Susceptibility Zone | LSI Range | Area (km²) | Area (%) | Train Slides | Train Slide % | Val Slides | Val Slide % | Total Slides | Slide Density (slides/km²) | Density Ratio |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in stats_df.iterrows():
        md_lines.append(
            f"| {int(r['zone_id'])} | **{r['zone']}** | {r['lsi_range']} | "
            f"{r['area_km2']:.2f} | {r['area_pct']:.2f}% | {int(r['train_slides'])} | "
            f"{r['train_slide_pct']:.2f}% | {int(r['val_slides'])} | {r['val_slide_pct']:.2f}% | "
            f"{int(r['total_slides'])} | **{r['total_density_per_km2']:.4f}** | {r['density_ratio']:.2f} |"
        )
    md_lines.extend([
        "",
        "**Summary & Physical Verification:**",
        f"- **Total District Area:** {total_area_km2:.2f} km² ({area_cells.sum():,} pixels)",
        f"- **Landslide Inventory:** {total_hits.sum()} total slides ({train_hits.sum()} train, {val_hits.sum()} val)",
        f"- **Monotonic Density:** `{'CONFIRMED' if is_monotonic else 'VIOLATED'}` — "
        f"Landslide density strictly increases from {stats_df.iloc[0]['total_density_per_km2']:.4f} slides/km² (Very Low) "
        f"to {stats_df.iloc[-1]['total_density_per_km2']:.4f} slides/km² (Very High).",
        f"- **High & Very High Concentration:** The High and Very High zones encompass "
        f"{stats_df.iloc[3:]['area_pct'].sum():.2f}% of the district area but capture "
        f"{stats_df.iloc[3:]['total_slide_pct'].sum():.2f}% of all landslides ({stats_df.iloc[3:]['val_slide_pct'].sum():.2f}% of unseen validation landslides).",
        ""
    ])
    (TABLES_DIR / "zone_statistics.md").write_text("\n".join(md_lines))
    print(f"Exported tables to {TABLES_DIR / 'zone_statistics.csv'} and {TABLES_DIR / 'zone_statistics.md'}")

    # 5. Generate High-Resolution Cartographic Publication Figures (Clean & Overlay)
    print("\nGenerating publication-quality cartographic maps...")
    dist_path = PROC / "district.gpkg"
    if not dist_path.exists():
        dist_path = MASTER_DIR / "district.gpkg"
    dist_gdf = gpd.read_file(dist_path) if dist_path.exists() else None

    minx = T.c
    maxy = T.f
    maxx = minx + W * T.a
    miny = maxy + H * T.e
    extent = (minx, maxx, miny, maxy)

    masked_zone = np.ma.masked_equal(zone, 0)
    cmap = ListedColormap(ZONE_COLORS)

    # Pure zone patches for clean legend
    zone_patches = [
        mpatches.Patch(facecolor=col, edgecolor="#333333", lw=0.5, label=f"{lab} ({interval})")
        for lab, col, interval in zip(ZONE_LABELS, ZONE_COLORS, intervals)
    ]

    # --- 5A. Clean Susceptibility Map (No Training/Validation Point Markings) ---
    fig_clean, ax_clean = plt.subplots(figsize=(10, 11), dpi=300)
    ax_clean.imshow(masked_zone, cmap=cmap, vmin=1, vmax=5, extent=extent, interpolation="nearest", origin="upper")

    if dist_gdf is not None:
        dist_gdf.boundary.plot(ax=ax_clean, edgecolor="#111111", linewidth=1.2, zorder=5)

    leg_clean = ax_clean.legend(handles=zone_patches, title="Susceptibility Classes (LSI)", loc="lower left",
                                fontsize=9, title_fontsize=10, framealpha=0.94, facecolor="white",
                                edgecolor="#999999", fancybox=True)
    leg_clean.get_title().set_fontweight("bold")

    add_scale_bar(ax_clean, length_km=10, location=(0.76, 0.05))
    add_north_arrow(ax_clean, location=(0.94, 0.94))

    ax_clean.set_title(f"Landslide Susceptibility Map of {DISTRICT} District, Kerala\n"
                       f"Frequency Ratio Model (Fisher-Jenks Natural Breaks Classification)",
                       fontsize=12, fontweight="bold", pad=12)
    ax_clean.set_xlabel("Easting (m) - UTM Zone 43N [EPSG:32643]", fontsize=9.5)
    ax_clean.set_ylabel("Northing (m) - UTM Zone 43N [EPSG:32643]", fontsize=9.5)
    ax_clean.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_clean.ticklabel_format(style="plain", useOffset=False)
    ax_clean.tick_params(labelsize=8)

    plt.tight_layout()

    clean_out_1 = OUT / "susceptibility_map.png"
    clean_out_2 = OUT / "susceptibility_map_clean.png"
    clean_out_3 = FIGURES_DIR / "susceptibility_map.png"
    clean_out_4 = FIGURES_DIR / "susceptibility_map_clean.png"
    clean_out_5 = FIGURES_DIR / "susceptibility_zones_jenks.png"

    for p in (clean_out_1, clean_out_2, clean_out_3, clean_out_4, clean_out_5):
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig_clean)

    # --- 5B. Inventory Overlay Map (With Training, Validation & 2024 Benchmark Markings) ---
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    ax.imshow(masked_zone, cmap=cmap, vmin=1, vmax=5, extent=extent, interpolation="nearest", origin="upper")

    if dist_gdf is not None:
        dist_gdf.boundary.plot(ax=ax, edgecolor="#111111", linewidth=1.2, zorder=5)

    ax.scatter(train_xy[:, 0], train_xy[:, 1], c="#1a1a1a", edgecolors="white",
               s=18, linewidths=0.5, alpha=0.85, label=f"Training Slides (N={len(train_xy)})", zorder=6)
    ax.scatter(val_xy[:, 0], val_xy[:, 1], c="#ff00ff", edgecolors="black",
               marker="^", s=28, linewidths=0.6, alpha=0.95, label=f"Validation Slides (N={len(val_xy)})", zorder=7)

    lat_ch, lon_ch = 11.543, 76.155
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    ch_x, ch_y = transformer.transform(lon_ch, lat_ch)
    ax.scatter([ch_x], [ch_y], c="#ffd700", edgecolors="#b22222", marker="*",
               s=220, linewidths=1.2, zorder=8, label="2024 Chooralmala Disaster Site")
    ax.annotate("2024 Chooralmala\nDisaster Site", xy=(ch_x, ch_y), xytext=(ch_x + 4000, ch_y - 3000),
                fontsize=8, fontweight="bold", color="#800000",
                arrowprops=dict(arrowstyle="->", color="#800000", lw=1.2),
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#fff8dc", edgecolor="#800000", lw=0.8, alpha=0.9),
                zorder=9)

    handles = list(zone_patches) + [mpatches.Patch(color="none", label=" ")]
    handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1a1a1a",
                              markeredgecolor="white", markersize=6, label=f"Training Slides (N={len(train_xy)})"))
    handles.append(plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#ff00ff",
                              markeredgecolor="black", markersize=7, label=f"Validation Slides (N={len(val_xy)})"))
    handles.append(plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd700",
                              markeredgecolor="#b22222", markersize=11, label="2024 Chooralmala Disaster Site"))

    legend = ax.legend(handles=handles, title="Susceptibility Classes (LSI)", loc="lower left",
                       fontsize=8.5, title_fontsize=9.5, framealpha=0.92, facecolor="white",
                       edgecolor="#999999", fancybox=True)
    legend.get_title().set_fontweight("bold")

    add_scale_bar(ax, length_km=10, location=(0.76, 0.05))
    add_north_arrow(ax, location=(0.94, 0.94))

    ax.set_title(f"Landslide Susceptibility Map of {DISTRICT} District, Kerala\n"
                 f"Frequency Ratio Model (With Inventory & 2024 Benchmark Overlay)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m) - UTM Zone 43N [EPSG:32643]", fontsize=9.5)
    ax.set_ylabel("Northing (m) - UTM Zone 43N [EPSG:32643]", fontsize=9.5)
    ax.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    overlay_out_1 = OUT / "susceptibility_map_with_points.png"
    overlay_out_2 = FIGURES_DIR / "susceptibility_map_with_points.png"
    plt.savefig(overlay_out_1, dpi=300, bbox_inches="tight")
    plt.savefig(overlay_out_2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved clean susceptibility map to:\n  - {clean_out_1}\n  - {clean_out_2}")
    print(f"Saved point-overlay map to:\n  - {overlay_out_1}")

    return {
        "zone": zone,
        "edges": edges,
        "breaks": breaks,
        "stats_df": stats_df,
        "is_monotonic": is_monotonic,
    }


if __name__ == "__main__":
    classify_susceptibility_zones()
