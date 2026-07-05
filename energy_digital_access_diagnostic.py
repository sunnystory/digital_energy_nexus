# -*- coding: utf-8 -*-
"""
Energy & Digital Access Diagnostic — Public Institutions (Liberia example)

Pipeline:
    1. Download education + health public institutions from OpenStreetMap.
    2. Load existing/planned energy infrastructure layers (grid line,
       transformers, planned NEA expansion transformers, mini-grids).
    3. Load telecom tower data and classify each institution's mobile
       coverage using per-tower buffer radii.
    4. Classify each institution's energy access status.
    5. Combine energy + digital status into a four-way access typology:
       "Energy + Digital access", "Energy only", "Digital only",
       "No energy or digital access".
    6. Map and export the results.

Run as a script, or cell-by-cell in VSCode / Jupyter using the "# %%" markers.

Requires: geopandas, osmnx, shapely, folium, pyogrio, pandas, numpy
    pip install geopandas osmnx shapely folium pyogrio pandas numpy openpyxl
"""

# %%
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from folium.plugins import MarkerCluster

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        pass

# =========================================================
# 1. CONFIGURATION
# =========================================================

PLACE_NAME = "Liberia"
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32629"  # UTM Zone 29N, appropriate for Liberia

DOWNLOADS_DIR = r"C:\Users\wb632724\Downloads"
OUTPUT_DIR = os.path.join(DOWNLOADS_DIR, "e&d", "outputs", "public_institution_access")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Energy infrastructure layers (existing + planned) ---
ENERGY_DIR = os.path.join(DOWNLOADS_DIR, "energy_data_liberia")
DIST_LINE_PATH = os.path.join(ENERGY_DIR, "liberia-electric-distribution-line", "distribution_line.shp")
DIST_TRANSFORMER_PATH = os.path.join(ENERGY_DIR, "liberia-electric-distribution-transformers", "distribution_transformers.shp")
MINIGRID_PATH = os.path.join(ENERGY_DIR, "liberia-small-mini-grid-nea-project", "nea_small_minigrid.shp")

# No NEA grid-expansion-transformer layer is available for this Liberia run.
# Set this to a shapefile path to re-enable the "planned grid expansion" rule
# for a country/dataset that has one.
NEA_EXPANSION_TRANSFORMER_PATH = None

# --- Digital tower data ---
TOWER_CSV_PATH = os.path.join(DOWNLOADS_DIR, "all_matched_sites_sharada (1).csv")
TOWER_LON_COL = "Longitude"
TOWER_LAT_COL = "Latitude"
TOWER_TECH_COL = "Technology"

# --- Optional Giga school data (set to a path to enable) ---
GIGA_PATH = None

# --- Energy access thresholds (km) ---
# "Existing" infrastructure (a, b, d) counts toward current energy access.
# NEA expansion transformers (c) are *planned*, not yet built, so they are
# tracked separately as a forward-looking flag rather than folded into the
# "energy access" typology used below.
DIST_LINE_THRESHOLD_KM = 5           # (a) existing distribution line
DIST_TRANSFORMER_THRESHOLD_KM = 5    # (b) existing distribution transformer
NEA_EXPANSION_THRESHOLD_KM = 5       # (c) planned NEA grid expansion transformer
MINIGRID_THRESHOLD_KM = 2            # (d) mini-grid

# --- Digital coverage buffer radius (km), by best technology at a tower ---
# The tower CSV's Technology field lists every generation present at a site
# (e.g. "2G/3G/4G", "2G/4G", "2G/3G", "2G"). The buffer radius follows the
# best generation present at that site:
DIGITAL_RADIUS_4G_KM = 8   # site includes 4G  (2G/3G/4G, 2G/4G)
DIGITAL_RADIUS_3G_KM = 10  # site includes 3G but not 4G (2G/3G)
DIGITAL_RADIUS_2G_KM = 10  # site is 2G only (2G)

# Uncomment to use an alternate Overpass mirror if the default endpoint
# times out on country-wide queries.
# ox.settings.overpass_url = "https://overpass.private.coffee/api"

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 600


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def make_point_gdf_from_lonlat(df, lon_col="Longitude", lat_col="Latitude", crs=CRS_WGS84):
    """Convert a DataFrame with longitude/latitude columns into a point GeoDataFrame."""
    df = df.copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs)


def to_representative_points(gdf):
    """Collapse any geometry type (polygon, line, multi-*) to a single representative point."""
    gdf = gdf.copy()
    gdf_m = gdf.to_crs(CRS_METRIC)
    gdf_m["geometry"] = gdf_m.geometry.representative_point()
    return gdf_m.to_crs(CRS_WGS84)


def read_vector(path, layer_name=None):
    """Read a vector file and force it to WGS84."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        print(f"Warning: {layer_name or path} has no CRS. Assuming {CRS_WGS84}.")
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)
    gdf["source_layer"] = layer_name or os.path.basename(path)
    return gdf


def add_nearest_distance_km(points_gdf, target_gdf, distance_col):
    """Add the distance (km) from each point to the nearest feature in target_gdf."""
    points = points_gdf.copy().reset_index(drop=True)
    points["pid"] = points.index

    if target_gdf is None or len(target_gdf) == 0:
        points[distance_col] = np.nan
        return points.drop(columns=["pid"])

    points_m = points.to_crs(CRS_METRIC)
    target_m = target_gdf.to_crs(CRS_METRIC)
    target_m = target_m[target_m.geometry.notna()]

    joined = gpd.sjoin_nearest(
        points_m[["pid", "geometry"]],
        target_m[["geometry"]],
        how="left",
        distance_col="dist_m",
    )
    nearest_km = joined.groupby("pid")["dist_m"].min() / 1000
    points[distance_col] = points["pid"].map(nearest_km)
    return points.drop(columns=["pid"])


def get_map_center(*gdfs):
    """Get a reasonable map center from one or more GeoDataFrames."""
    frames = [g.to_crs(CRS_WGS84) for g in gdfs if g is not None and len(g) > 0]
    if not frames:
        return [6.4281, -9.4295]  # Liberia, approximate center
    combined = pd.concat(frames, ignore_index=True)
    return [combined.geometry.y.mean(), combined.geometry.x.mean()]


def save_map(m, filename):
    """Save a Folium map to OUTPUT_DIR and display it if running in a notebook."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    m.save(out_path)
    print(f"Saved map: {out_path}")
    display(m)
    return out_path


# %%
# =========================================================
# 3. DOWNLOAD PUBLIC INSTITUTIONS (EDUCATION + HEALTH) FROM OSM
# =========================================================

EDUCATION_TAGS = {
    "amenity": ["school", "kindergarten", "college", "university", "library"]
}

HEALTH_TAGS = {
    # Remove "pharmacy"/"dentist" below if they should not count as health institutions.
    "amenity": ["hospital", "clinic", "doctors", "health_post", "pharmacy", "dentist"],
    "healthcare": True,
}


def download_osm_institutions(place_name, tags, institution_type, max_retries=3, retry_wait_sec=10):
    """Download one institution category (Education/Health) country-wide from OSM, with retries."""
    print(f"\nDownloading OSM institutions: {institution_type}")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  attempt {attempt}/{max_retries}")
            gdf = ox.features_from_place(place_name, tags=tags)
            gdf = gdf.reset_index()
            gdf = gdf[gdf.geometry.notna()].copy()
            gdf["institution_type"] = institution_type
            print(f"  success: {len(gdf):,} features")
            return gdf
        except Exception as exc:
            print(f"  failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(retry_wait_sec)


education_raw = download_osm_institutions(PLACE_NAME, EDUCATION_TAGS, "Education")
health_raw = download_osm_institutions(PLACE_NAME, HEALTH_TAGS, "Health")

osm_raw = pd.concat([education_raw, health_raw], ignore_index=True)
osm_raw = gpd.GeoDataFrame(osm_raw, geometry="geometry", crs=CRS_WGS84)

# Deduplicate OSM objects (a feature can be returned by more than one tag match)
dedup_cols = [c for c in ["element_type", "osmid"] if c in osm_raw.columns]
osm_raw = osm_raw.drop_duplicates(subset=dedup_cols) if len(dedup_cols) == 2 else osm_raw.drop_duplicates()

print("Combined raw OSM institutions:", osm_raw.shape)

# Convert all geometries (points, polygons, lines) to a single representative point
osm_points = to_representative_points(osm_raw)

if "name" not in osm_points.columns:
    osm_points["name"] = None
osm_points["institution_name"] = osm_points["name"].fillna("Unnamed OSM institution")
osm_points["institution_source"] = "OSM"

public_inst = osm_points[
    ["institution_name", "institution_type", "institution_source", "geometry"]
].reset_index(drop=True)
public_inst["institution_id"] = public_inst.index + 1

print("Public institutions (Education + Health):", public_inst.shape)
print(public_inst["institution_type"].value_counts())


# %%
# =========================================================
# 3b. OPTIONAL: ADD GIGA SCHOOL DATA
# =========================================================

def detect_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


if GIGA_PATH is not None and os.path.exists(GIGA_PATH):
    giga_df = pd.read_csv(GIGA_PATH) if GIGA_PATH.lower().endswith(".csv") else pd.read_excel(GIGA_PATH)

    lon_col = detect_column(giga_df.columns, ["longitude", "lon", "x"])
    lat_col = detect_column(giga_df.columns, ["latitude", "lat", "y"])
    name_col = detect_column(giga_df.columns, ["name", "school_name", "school"])
    if lon_col is None or lat_col is None:
        raise ValueError("Could not detect longitude/latitude columns in Giga data.")

    giga_gdf = make_point_gdf_from_lonlat(giga_df, lon_col=lon_col, lat_col=lat_col)
    giga_inst = gpd.GeoDataFrame({
        "institution_name": giga_gdf[name_col] if name_col else "Giga school",
        "institution_type": "Education",
        "institution_source": "Giga",
        "geometry": giga_gdf.geometry,
    }, crs=CRS_WGS84)

    public_inst = pd.concat([public_inst, giga_inst], ignore_index=True)
    public_inst = gpd.GeoDataFrame(public_inst, geometry="geometry", crs=CRS_WGS84)
    public_inst["institution_id"] = public_inst.index + 1
    print("Combined public institutions (OSM + Giga):", public_inst.shape)
else:
    print("No Giga file provided. Using OSM institutions only.")


# %%
# =========================================================
# 4. LOAD ENERGY INFRASTRUCTURE LAYERS
# =========================================================

dist_line = read_vector(DIST_LINE_PATH, "Existing distribution line")
dist_transformers = read_vector(DIST_TRANSFORMER_PATH, "Existing distribution transformer")
minigrid = read_vector(MINIGRID_PATH, "Mini-grid")

# Optional: only present when NEA_EXPANSION_TRANSFORMER_PATH is configured.
nea_expansion_transformers = (
    read_vector(NEA_EXPANSION_TRANSFORMER_PATH, "Planned NEA grid expansion transformer")
    if NEA_EXPANSION_TRANSFORMER_PATH else None
)

for name, gdf in [
    ("Distribution lines", dist_line),
    ("Distribution transformers", dist_transformers),
    ("NEA expansion transformers", nea_expansion_transformers),
    ("Mini-grids", minigrid),
]:
    if gdf is None:
        print(f"{name}: not available, skipped")
    else:
        print(f"{name}: {gdf.shape}, geometry types: {gdf.geom_type.value_counts().to_dict()}")


# %%
# =========================================================
# 5. LOAD AND CLASSIFY DIGITAL TOWER DATA
# =========================================================

tower_df = pd.read_csv(TOWER_CSV_PATH)
towers = make_point_gdf_from_lonlat(tower_df, lon_col=TOWER_LON_COL, lat_col=TOWER_LAT_COL, crs=CRS_WGS84)


def classify_tower_technology(tech):
    """Map a tower's technology string to a coverage class and buffer radius (km)."""
    tech = str(tech).upper()
    if "4G" in tech:
        return "4G", DIGITAL_RADIUS_4G_KM
    if "3G" in tech:
        return "3G", DIGITAL_RADIUS_3G_KM
    if "2G" in tech:
        return "2G", DIGITAL_RADIUS_2G_KM
    return "Unknown", np.nan


towers[["tower_tech_class", "tower_radius_km"]] = towers[TOWER_TECH_COL].apply(
    lambda x: pd.Series(classify_tower_technology(x))
)
towers = towers.dropna(subset=["tower_radius_km"]).copy()  # drops "NA"/unclassifiable rows
towers["tower_id"] = range(1, len(towers) + 1)

print("Tower coverage classes:")
print(towers["tower_tech_class"].value_counts())

towers_4g = towers[towers["tower_tech_class"] == "4G"]
towers_3g = towers[towers["tower_tech_class"] == "3G"]
towers_2g = towers[towers["tower_tech_class"] == "2G"]


# %%
# =========================================================
# 6. ENERGY ACCESSIBILITY ANALYSIS
# =========================================================

inst_access = public_inst.copy().reset_index(drop=True)
inst_access["institution_id"] = inst_access.index + 1

inst_access = add_nearest_distance_km(inst_access, dist_line, "dist_to_distribution_line_km")
inst_access = add_nearest_distance_km(inst_access, dist_transformers, "dist_to_distribution_transformer_km")
inst_access = add_nearest_distance_km(inst_access, nea_expansion_transformers, "dist_to_nea_grid_expansion_transformer_km")
inst_access = add_nearest_distance_km(inst_access, minigrid, "dist_to_minigrid_km")

inst_access["near_distribution_line_5km"] = inst_access["dist_to_distribution_line_km"] <= DIST_LINE_THRESHOLD_KM
inst_access["near_distribution_transformer_5km"] = inst_access["dist_to_distribution_transformer_km"] <= DIST_TRANSFORMER_THRESHOLD_KM
inst_access["near_nea_grid_expansion_transformer_5km"] = inst_access["dist_to_nea_grid_expansion_transformer_km"] <= NEA_EXPANSION_THRESHOLD_KM
inst_access["near_minigrid_2km"] = inst_access["dist_to_minigrid_km"] <= MINIGRID_THRESHOLD_KM

# Current energy access = existing infrastructure only (a, b, d).
# Planned NEA expansion (c) is tracked separately, see "energy_status" below.
inst_access["energy_access_proxy"] = (
    inst_access["near_distribution_transformer_5km"]
    | inst_access["near_minigrid_2km"]
    | inst_access["near_distribution_line_5km"]
)

energy_status_conditions = [
    inst_access["near_distribution_transformer_5km"],
    inst_access["near_minigrid_2km"],
    inst_access["near_distribution_line_5km"],
    inst_access["near_nea_grid_expansion_transformer_5km"],
]
energy_status_choices = [
    "Likely grid-covered: existing transformer within 5 km",
    "Likely mini-grid-covered: mini-grid within 2 km",
    "Likely grid-accessible: existing distribution line within 5 km",
    "Planned/proposed grid expansion area: NEA transformer within 5 km",
]
inst_access["energy_status"] = np.select(energy_status_conditions, energy_status_choices, default="No energy access proxy identified")


# %%
# =========================================================
# 7. DIGITAL ACCESSIBILITY ANALYSIS
# =========================================================

inst_access = add_nearest_distance_km(inst_access, towers_4g, "dist_to_nearest_4g_tower_km")
inst_access = add_nearest_distance_km(inst_access, towers_3g, "dist_to_nearest_3g_tower_km")
inst_access = add_nearest_distance_km(inst_access, towers_2g, "dist_to_nearest_2g_tower_km")

inst_access["has_4g_coverage_proxy"] = inst_access["dist_to_nearest_4g_tower_km"] <= DIGITAL_RADIUS_4G_KM
inst_access["has_3g_coverage_proxy"] = inst_access["dist_to_nearest_3g_tower_km"] <= DIGITAL_RADIUS_3G_KM
inst_access["has_2g_coverage_proxy"] = inst_access["dist_to_nearest_2g_tower_km"] <= DIGITAL_RADIUS_2G_KM

inst_access["digital_access_proxy"] = (
    inst_access["has_4g_coverage_proxy"]
    | inst_access["has_3g_coverage_proxy"]
    | inst_access["has_2g_coverage_proxy"]
)

digital_status_conditions = [
    inst_access["has_4g_coverage_proxy"],
    inst_access["has_3g_coverage_proxy"],
    inst_access["has_2g_coverage_proxy"],
]
digital_status_choices = ["4G", "3G", "2G"]
inst_access["digital_status"] = np.select(digital_status_conditions, digital_status_choices, default="No mobile coverage proxy identified")


# %%
# =========================================================
# 8. COMBINED ENERGY + DIGITAL ACCESS TYPOLOGY
# =========================================================

combined_conditions = [
    inst_access["energy_access_proxy"] & inst_access["digital_access_proxy"],
    inst_access["energy_access_proxy"] & ~inst_access["digital_access_proxy"],
    ~inst_access["energy_access_proxy"] & inst_access["digital_access_proxy"],
    ~inst_access["energy_access_proxy"] & ~inst_access["digital_access_proxy"],
]
combined_choices = [
    "Energy + Digital access",
    "Energy only",
    "Digital only",
    "No energy or digital access",
]
inst_access["combined_access_status"] = np.select(combined_conditions, combined_choices, default="Unclassified")

summary_by_type = (
    inst_access.groupby(["institution_type", "combined_access_status"])
    .size()
    .reset_index(name="n_institutions")
    .sort_values(["institution_type", "combined_access_status"])
)

overall_summary = (
    inst_access["combined_access_status"]
    .value_counts()
    .rename_axis("combined_access_status")
    .reset_index(name="n_institutions")
)
overall_summary["share_percent"] = (overall_summary["n_institutions"] / overall_summary["n_institutions"].sum() * 100).round(1)

print(summary_by_type)
print(overall_summary)


# %%
# =========================================================
# 9. MAPS
# =========================================================

# --- Public institutions ---
m_inst = folium.Map(location=get_map_center(public_inst), zoom_start=7, tiles="CartoDB positron")
inst_type_colors = {"Education": "blue", "Health": "red"}
inst_cluster = MarkerCluster(name="Public institutions").add_to(m_inst)
for _, row in public_inst.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color=inst_type_colors.get(row["institution_type"], "gray"),
        fill=True,
        fill_color=inst_type_colors.get(row["institution_type"], "gray"),
        fill_opacity=0.75,
        popup=folium.Popup(
            f"<b>{row['institution_name']}</b><br>Type: {row['institution_type']}<br>Source: {row['institution_source']}",
            max_width=300,
        ),
    ).add_to(inst_cluster)
folium.LayerControl(collapsed=False).add_to(m_inst)
save_map(m_inst, "01_public_institutions.html")

# --- Energy infrastructure ---
center = get_map_center(public_inst, dist_line, dist_transformers, nea_expansion_transformers, minigrid)
m_energy = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

folium.GeoJson(
    dist_line, name="Existing distribution line",
    style_function=lambda x: {"color": "black", "weight": 2, "opacity": 0.8},
).add_to(m_energy)

energy_point_layers = [
    (dist_transformers, "Existing distribution transformers", "red"),
    (nea_expansion_transformers, "NEA grid expansion transformers (planned)", "orange"),
    (minigrid, "Mini-grids", "green"),
]
for layer_gdf, layer_name, color in energy_point_layers:
    if layer_gdf is None:
        continue
    cluster = MarkerCluster(name=layer_name).add_to(m_energy)
    for _, row in layer_gdf.iterrows():
        geom = row.geometry if row.geometry.geom_type == "Point" else row.geometry.representative_point()
        folium.CircleMarker(
            location=[geom.y, geom.x], radius=4, color=color, fill=True,
            fill_color=color, fill_opacity=0.8, popup=layer_name,
        ).add_to(cluster)
folium.LayerControl(collapsed=False).add_to(m_energy)
save_map(m_energy, "02_energy_layers.html")

# --- Digital towers ---
center = get_map_center(public_inst, towers)
m_towers = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
tower_colors = {"4G": "green", "3G": "orange", "2G": "blue"}
tower_cluster = MarkerCluster(name="Telecom towers").add_to(m_towers)
for _, row in towers.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color=tower_colors.get(row["tower_tech_class"], "gray"),
        fill=True,
        fill_color=tower_colors.get(row["tower_tech_class"], "gray"),
        fill_opacity=0.75,
        popup=f"Tower {row['tower_id']}<br>Class: {row['tower_tech_class']}<br>Buffer: {row['tower_radius_km']} km",
    ).add_to(tower_cluster)
folium.LayerControl(collapsed=False).add_to(m_towers)
save_map(m_towers, "03_digital_towers.html")

# --- Final combined access typology ---
center = get_map_center(inst_access)
m_final = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
access_colors = {
    "Energy + Digital access": "green",
    "Energy only": "orange",
    "Digital only": "blue",
    "No energy or digital access": "red",
}
final_cluster = MarkerCluster(name="Public institutions by combined access status").add_to(m_final)
for _, row in inst_access.iterrows():
    popup = f"""
    <b>{row['institution_name']}</b><br>
    Type: {row['institution_type']}<br><br>
    <b>Combined status:</b> {row['combined_access_status']}<br>
    <b>Energy status:</b> {row['energy_status']}<br>
    <b>Digital status:</b> {row['digital_status']}<br><br>
    Dist. to distribution line: {row['dist_to_distribution_line_km']:.2f} km<br>
    Dist. to distribution transformer: {row['dist_to_distribution_transformer_km']:.2f} km<br>
    Dist. to NEA expansion transformer: {row['dist_to_nea_grid_expansion_transformer_km']:.2f} km<br>
    Dist. to mini-grid: {row['dist_to_minigrid_km']:.2f} km<br><br>
    Dist. to 4G tower: {row['dist_to_nearest_4g_tower_km']:.2f} km<br>
    Dist. to 3G tower: {row['dist_to_nearest_3g_tower_km']:.2f} km<br>
    Dist. to 2G tower: {row['dist_to_nearest_2g_tower_km']:.2f} km
    """
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color=access_colors.get(row["combined_access_status"], "gray"),
        fill=True,
        fill_color=access_colors.get(row["combined_access_status"], "gray"),
        fill_opacity=0.75,
        popup=folium.Popup(popup, max_width=400),
    ).add_to(final_cluster)

legend_html = """
<div style="position: fixed; bottom: 40px; left: 40px; width: 260px;
background-color: white; z-index:9999; font-size:13px; border:2px solid grey; padding: 10px;">
<b>Public Institution Access Status</b><br><br>
<span style="color:green;">&#9679;</span> Energy + Digital access<br>
<span style="color:orange;">&#9679;</span> Energy only<br>
<span style="color:blue;">&#9679;</span> Digital only<br>
<span style="color:red;">&#9679;</span> No energy or digital access<br>
</div>
"""
m_final.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m_final)
save_map(m_final, "04_public_institution_energy_digital_access.html")


# %%
# =========================================================
# 10. SAVE OUTPUTS
# =========================================================

gpkg_path = os.path.join(OUTPUT_DIR, "public_institution_energy_digital_access.gpkg")
inst_access.to_file(gpkg_path, layer="institution_access", driver="GPKG")

inst_csv = inst_access.copy()
inst_csv["longitude"] = inst_csv.geometry.x
inst_csv["latitude"] = inst_csv.geometry.y
inst_csv = pd.DataFrame(inst_csv.drop(columns="geometry"))
csv_path = os.path.join(OUTPUT_DIR, "public_institution_energy_digital_access.csv")
inst_csv.to_csv(csv_path, index=False)

summary_path = os.path.join(OUTPUT_DIR, "summary_by_institution_type.csv")
overall_summary_path = os.path.join(OUTPUT_DIR, "summary_overall.csv")
summary_by_type.to_csv(summary_path, index=False)
overall_summary.to_csv(overall_summary_path, index=False)

print("Saved:")
print(gpkg_path)
print(csv_path)
print(summary_path)
print(overall_summary_path)

# `inst_access` (one row per institution, with combined_access_status,
# energy_status, digital_status, and all distance columns) is the input for
# the next stage: joining nearby settlement population / security-risk data
# to assess co-deployment / procurement viability (e.g. "Digital only" +
# high population + low risk => viable for new energy investment;
# "No energy or digital access" + low population + high risk => deprioritize).
