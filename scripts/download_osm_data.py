"""Download and prepare OSM roads and waterways for Wayanad District.
Fetches line geometries from Overpass API and clips to the district boundary.
"""
import requests
import json
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OSM_DIR = DATA_DIR / "osm"
OSM_DIR.mkdir(parents=True, exist_ok=True)

boundary_gdf = gpd.read_file(DATA_DIR / "boundary" / "wayanad_boundary.geojson")
boundary_poly = boundary_gdf.geometry.iloc[0]
bbox = "11.45,75.75,12.00,76.45"
headers = {"User-Agent": "WayanadResearchProject/1.0 (contact: research@wayanad.org)"}
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

def fetch_and_save_ways(query, output_path, name):
    if output_path.exists():
        print(f"{output_path} already exists, skipping.")
        return

    print(f"Fetching {name} from Overpass API...")
    data = None
    for ep in ENDPOINTS:
        try:
            print(f"Trying endpoint: {ep}")
            r = requests.post(ep, data={"data": query}, headers=headers, timeout=120)
            if r.status_code == 200:
                data = r.json()
                break
            else:
                print(f"Endpoint {ep} returned status {r.status_code}")
        except Exception as e:
            print(f"Endpoint {ep} failed: {e}")

    if not data:
        raise RuntimeError(f"All Overpass endpoints failed for {name}")

    elements = data.get("elements", [])
    print(f"Retrieved {len(elements)} raw {name} elements.")
    
    features = []
    for elem in elements:
        geom_points = elem.get("geometry", [])
        if len(geom_points) >= 2:
            coords = [(pt["lon"], pt["lat"]) for pt in geom_points]
            line = LineString(coords)
            tags = elem.get("tags", {})
            tags["osm_id"] = elem.get("id")
            tags["geometry"] = line
            features.append(tags)
            
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    print(f"Built GeoDataFrame with {len(gdf)} {name} lines.")
    
    # Clip to district boundary and filter out non-linear artifacts (e.g. boundary tangent points)
    clipped = gpd.clip(gdf, boundary_gdf)
    clipped = clipped[clipped.geom_type.isin(["LineString", "MultiLineString"])]
    print(f"Clipped {name} within Wayanad: {len(clipped)} linear segments.")
    clipped.to_file(output_path, driver="GeoJSON")
    print(f"Saved to {output_path}")

# 1. Roads Query
q_roads = f"""
[out:json][timeout:90];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]({bbox});
);
out geom;
"""
fetch_and_save_ways(q_roads, OSM_DIR / "wayanad_roads.geojson", "roads")

# 2. Waterways Query
q_waterways = f"""
[out:json][timeout:90];
(
  way["waterway"~"river|stream|canal|drain"]({bbox});
);
out geom;
"""
fetch_and_save_ways(q_waterways, OSM_DIR / "wayanad_streams.geojson", "streams")
