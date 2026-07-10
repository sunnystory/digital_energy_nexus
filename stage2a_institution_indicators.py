# -*- coding: utf-8 -*-
"""
Stage 2a — Institution-Level Delivery-Matching Indicators (Liberia)

For every public institution in the Stage 1 output
(public_institution_energy_digital_access.gpkg, layer "institution_access"),
computes buffer zonal statistics / nearest-distance indicators around the
institution's own point location. This replaces the settlement-attribute
join previously used for Market Viability Index (MVI) / Physical Constraint
Index (PCI) scoring in Stage 2 with per-institution demand and supply
indicators.

Output: institution_indicators.gpkg (layer "indicators") + matching CSV,
under C:\\Users\\wb632724\\Downloads\\e&d\\outputs\\delivery_matching\\.

Output columns
--------------
Demand-side:
    pop_2km, pop_10km          WorldPop Global 2015-2030 R2024B constrained
                                100m, 2030 (primary), or WorldPop
                                Global_2000_2020 1km UNadj, 2020 (fallback).
    pop_source                  which WorldPop product was used.
    builtup_m2_2km              GHSL GHS-BUILT-S E2030 R2023A 100m
                                (Mollweide), 2030 epoch.
    n_buildings_2km,
    mean_building_area_2km      Google Open Buildings v3 (points), 2023
                                vintage imagery, S2 level-6 tiles.
    mean_building_height_2km    WSF3D building height raster (vintage per
                                config URL; NaN if unavailable).
    rwi_popweighted_10km,
    rwi_method                  Meta Relative Wealth Index (2019-2021),
                                population-weighted using WorldPop.
    n_other_institutions_10km   Stage 1 institutions gpkg (OSM + Giga).
Supply-side:
    dist_hub_km                 OSM place=city|town, current OSM snapshot.
    dist_main_road_km           OSM highway=motorway/trunk/primary/secondary.
    dist_any_road_km,
    no_road_2km, no_road_5km    OSM highway, drivable classes (config list).
    dist_transmission_km        Local shapefile: Liberia Electricity
                                Transmission Network.
    flood_rp100                  JRC Global River Flood Hazard Maps, RP100y.
    landslide_precip             GFDRR rainfall-triggered landslide hazard,
                                mean 1980-2018.
    security_risk_fallback,
    security_match_km           DRE Atlas nearest-settlement security_risk
                                (optional fallback, USE_SECURITY=True only).
Cross-check only (never used in scoring):
    dreatlas_pop_nearest, dreatlas_demand_nearest, dreatlas_match_km
                                DRE Atlas nearest-settlement population/demand.

Run as a script, or cell-by-cell in VSCode / Jupyter using the "# %%" markers.
"""

# %%
# On a corporate network, Python's bundled CA list (certifi) usually doesn't
# include the corporate proxy's TLS-inspection root certificate, so requests
# to nominatim/overpass/worldpop/etc fail with SSLCertVerificationError.
# `truststore` makes Python use the Windows certificate store instead (which
# already trusts the corporate root), so this must run before osmnx/requests
# make any HTTP calls.
import truststore
truststore.inject_into_ssl()

import os
import glob
import warnings
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import osmnx as ox
import rasterio
import rasterstats
import s2sphere
from shapely import wkt as _wkt
from shapely.geometry import box

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
CRS_METRIC = "EPSG:32629"  # UTM Zone 29N

DOWNLOADS_DIR = r"C:\Users\wb632724\Downloads"
BASE_DIR = os.path.join(DOWNLOADS_DIR, "e&d")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "delivery_matching")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE1_GPKG = os.path.join(
    BASE_DIR, "outputs", "public_institution_access",
    "public_institution_energy_digital_access.gpkg",
)
STAGE1_LAYER = "institution_access"

# --- Buffer radii (km) ---
R_COMMUNITY_KM = 2    # near-field demand / building / no-road buffer
R_CATCHMENT_KM = 10   # wider demand catchment (population, RWI, bundling)
R_NEAR_ROAD_KM = 2
R_FAR_ROAD_KM = 5
R_HAZARD_KM = 1       # flood / landslide buffer

# --- OSM road classes ---
MAIN_ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary"]
DRIVABLE_CLASSES = [
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "unclassified", "track",
]

# --- Local energy infrastructure shapefile ---
TRANSMISSION_DIR = os.path.join(
    DOWNLOADS_DIR, "energy_data_liberia", "liberia-electricity-transmission-network"
)

# --- WorldPop population raster ---
# Primary: Global 2015-2030 R2024B, constrained, 100m, 2030 projection.
WORLDPOP_2030_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2030/"
    "LBR/v1/100m/constrained/lbr_pop_2030_CN_100m_R2024B_v1.tif"
)
# Fallback: Global_2000_2020, 2020, 1km, UN-adjusted.
WORLDPOP_FALLBACK_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/LBR/"
    "lbr_ppp_2020_UNadj.tif"
)

# --- GHSL BUILT-S (Mollweide, ESRI:54009) ---
CRS_GHSL = "ESRI:54009"
GHSL_ZIP_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
    "GHS_BUILT_S_E2030_GLOBE_R2023A_54009_100/V1-0/"
    "GHS_BUILT_S_E2030_GLOBE_R2023A_54009_100_V1_0.zip"
)

# --- Google Open Buildings v3 (points, S2 level-6, no header) ---
BUILDING_SOURCE = "open_buildings"  # "open_buildings" | "wsf"
OPEN_BUILDINGS_S2_LEVEL = 6
OPEN_BUILDINGS_BASE_URL = (
    "https://storage.googleapis.com/open-buildings-data/v3/"
    "points_s2_level_6_gzip_no_header"
)
OPEN_BUILDINGS_COLUMNS = ["latitude", "longitude", "area_in_meters", "confidence", "full_plus_code"]

# --- WSF3D building height (placeholder; fill in when available) ---
WSF3D_BH_URL = None
SKIP_IF_UNAVAILABLE = True

# --- Meta Relative Wealth Index ---
# Not auto-downloadable; place the CSV (lat/lon/rwi grid, ~2.4km spacing) in
# CACHE_DIR yourself (e.g. from the RWI HDX page / worldbank RWI repo).
RWI_CSV_PATH = os.path.join(DOWNLOADS_DIR, "lbr_relative_wealth_index.csv")

# --- Hazard layers ---
JRC_FLOOD_ZIP_URL = "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp100y.zip"
GFDRR_LANDSLIDE_COG_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/1/DR0045418/LS_RF_Mean_1980-2018_COG.tif"

# --- DRE Atlas (fallback / cross-check only, never in the MVI/PCI score) ---
DRE_ATLAS_CSV = os.path.join(DOWNLOADS_DIR, "liberia_dre_atlas_settlements (1).csv")
USE_SECURITY = False

# --- Completeness reporting ---
NAN_WARNING_THRESHOLD_PCT = 20.0

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 600
ox.settings.max_query_area_size = 150_000_000_000


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def cached_download(url, dest_path, timeout=180, chunk_size=1 << 20):
    """Stream-download url to dest_path via requests, unless it already exists."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  cached: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return True
    print(f"  downloading: {url}")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp_path = dest_path + ".part"
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, dest_path)
        print(f"  saved: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as exc:
        print(f"  FAILED download {url}: {exc}")
        if os.path.exists(dest_path + ".part"):
            os.remove(dest_path + ".part")
        return False


def glob_first(dirpath, pattern="*.shp"):
    """Find the first file in dirpath matching pattern; print which one was picked."""
    matches = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {dirpath}")
    if len(matches) > 1:
        print(f"  multiple matches for {pattern} in {dirpath}, using first: {matches}")
    print(f"  using file: {matches[0]}")
    return matches[0]


def get_liberia_boundary():
    """Fetch (and cache) the Liberia country boundary polygon + WGS84 bbox."""
    cache_path = os.path.join(CACHE_DIR, "liberia_boundary.gpkg")
    if os.path.exists(cache_path):
        boundary = gpd.read_file(cache_path)
    else:
        boundary = ox.geocode_to_gdf(PLACE_NAME)
        boundary.to_file(cache_path, driver="GPKG")
        print(f"  cached boundary: {cache_path}")
    bbox = tuple(boundary.total_bounds)  # (minx, miny, maxx, maxy) in WGS84
    return boundary, bbox


def add_nearest_distance_km(points_gdf, target_gdf, distance_col):
    """Vectorized nearest-distance (km) from each point to any feature in target_gdf."""
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


def make_buffer_geoseries(points_gdf_metric, radius_km):
    """Buffer each point (already in a metric CRS) by radius_km, returns a GeoSeries."""
    return points_gdf_metric.geometry.buffer(radius_km * 1000)


def zonal_stat_over_buffers(buffers_wgs84, raster_path, stat="sum"):
    """
    Reproject buffer polygons to the raster's own CRS (never resample the
    raster itself) and run rasterstats zonal_stats with all_touched=True.
    Returns a numpy array aligned with buffers_wgs84's index.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

    buffers_raster_crs = gpd.GeoSeries(buffers_wgs84, crs=CRS_WGS84).to_crs(raster_crs)
    results = rasterstats.zonal_stats(
        buffers_raster_crs.geometry,
        raster_path,
        stats=[stat],
        all_touched=True,
        nodata=nodata,
    )
    return np.array([r[stat] if r[stat] is not None else np.nan for r in results])


def completeness_report(df, columns):
    """Print % non-null and min/median/max for the given columns."""
    print("\n--- Completeness report ---")
    rows = []
    for col in columns:
        if col not in df.columns:
            print(f"  {col}: MISSING FROM DATAFRAME")
            continue
        non_null_pct = 100 * df[col].notna().mean()
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            stats = f"min={numeric.min():.3g} median={numeric.median():.3g} max={numeric.max():.3g}"
        else:
            stats = "(non-numeric)"
        rows.append((col, non_null_pct, stats))
        print(f"  {col}: {non_null_pct:.1f}% non-null, {stats}")
        if non_null_pct < 100 - NAN_WARNING_THRESHOLD_PCT:
            print(f"  *** WARNING: '{col}' is {100 - non_null_pct:.1f}% NaN "
                  f"(> {NAN_WARNING_THRESHOLD_PCT}% threshold) - check source availability/coverage ***")
    return pd.DataFrame(rows, columns=["column", "pct_non_null", "stats"])


# %%
# =========================================================
# 3. LOAD STAGE 1 INSTITUTIONS + BUILD BUFFERS
# =========================================================

institutions = gpd.read_file(STAGE1_GPKG, layer=STAGE1_LAYER)
institutions = institutions.reset_index(drop=True)
institutions["institution_id"] = institutions["institution_id"].astype(int)
print(f"Loaded {len(institutions):,} institutions from Stage 1.")

institutions_m = institutions.to_crs(CRS_METRIC)
buffer_2km_m = make_buffer_geoseries(institutions_m, R_COMMUNITY_KM)
buffer_10km_m = make_buffer_geoseries(institutions_m, R_CATCHMENT_KM)
buffer_1km_m = make_buffer_geoseries(institutions_m, R_HAZARD_KM)

buffer_2km_wgs84 = gpd.GeoSeries(buffer_2km_m, crs=CRS_METRIC).to_crs(CRS_WGS84)
buffer_10km_wgs84 = gpd.GeoSeries(buffer_10km_m, crs=CRS_METRIC).to_crs(CRS_WGS84)
buffer_1km_wgs84 = gpd.GeoSeries(buffer_1km_m, crs=CRS_METRIC).to_crs(CRS_WGS84)

# Carries forward every Stage 1 column (energy_status, digital_status,
# combined_access_status, near_*_km flags, tower/energy distances, ...) so
# that Stage 2 can run purely from this one file, with no settlement join.
indicators = institutions.copy()


# %%
# =========================================================
# 4. LIBERIA BOUNDARY (used to clip/window rasters and cover S2 tiles)
# =========================================================

print("Fetching/caching Liberia boundary...")
liberia_boundary, liberia_bbox = get_liberia_boundary()
print(f"Liberia bbox (WGS84): {liberia_bbox}")


# %%
# =========================================================
# 5. SUPPLY: DISTANCE TO NEAREST HUB (OSM place=city|town)
# =========================================================

hub_cache_path = os.path.join(CACHE_DIR, "osm_hubs.gpkg")
if os.path.exists(hub_cache_path):
    hubs = gpd.read_file(hub_cache_path)
    print(f"Loaded cached hubs: {len(hubs)}")
else:
    print("Downloading OSM hubs (place=city|town)...")
    hubs_raw = ox.features_from_place(PLACE_NAME, tags={"place": ["city", "town"]})
    hubs_raw = hubs_raw.reset_index()
    hubs_raw = hubs_raw[hubs_raw.geometry.notna()].copy()
    hubs_m = hubs_raw.to_crs(CRS_METRIC)
    hubs_m["geometry"] = hubs_m.geometry.representative_point()
    hubs = hubs_m.to_crs(CRS_WGS84)
    keep_cols = [c for c in ["name", "place", "geometry"] if c in hubs.columns]
    hubs = hubs[keep_cols]
    hubs.to_file(hub_cache_path, driver="GPKG")
    print(f"Cached hubs: {hub_cache_path} ({len(hubs)} features)")

indicators = add_nearest_distance_km(indicators, hubs, "dist_hub_km")


# %%
# =========================================================
# 6. SUPPLY: DISTANCE TO NEAREST MAIN ROAD (OSM highway class subset)
# =========================================================

main_road_cache_path = os.path.join(CACHE_DIR, "osm_main_roads.gpkg")
if os.path.exists(main_road_cache_path):
    main_roads = gpd.read_file(main_road_cache_path)
    print(f"Loaded cached main roads: {len(main_roads)}")
else:
    print(f"Downloading OSM main roads {MAIN_ROAD_CLASSES}...")
    main_roads = ox.features_from_place(PLACE_NAME, tags={"highway": MAIN_ROAD_CLASSES})
    main_roads = main_roads.reset_index()
    main_roads = main_roads[main_roads.geometry.notna()].copy()
    main_roads = main_roads[["geometry"]]
    main_roads.to_file(main_road_cache_path, driver="GPKG")
    print(f"Cached main roads: {main_road_cache_path} ({len(main_roads)} features)")

indicators = add_nearest_distance_km(indicators, main_roads, "dist_main_road_km")


# %%
# =========================================================
# 7. SUPPLY: NO DRIVABLE ROAD WITHIN 2KM / 5KM (OSM highway, all drivable classes)
# =========================================================

drivable_cache_path = os.path.join(CACHE_DIR, "osm_drivable_roads.gpkg")
if os.path.exists(drivable_cache_path):
    drivable_roads = gpd.read_file(drivable_cache_path)
    print(f"Loaded cached drivable roads: {len(drivable_roads)}")
else:
    print(f"Downloading OSM drivable roads {DRIVABLE_CLASSES}...")
    drivable_roads = ox.features_from_place(PLACE_NAME, tags={"highway": DRIVABLE_CLASSES})
    drivable_roads = drivable_roads.reset_index()
    drivable_roads = drivable_roads[drivable_roads.geometry.notna()].copy()
    drivable_roads = drivable_roads[["geometry"]]
    drivable_roads.to_file(drivable_cache_path, driver="GPKG")
    print(f"Cached drivable roads: {drivable_cache_path} ({len(drivable_roads)} features)")

indicators = add_nearest_distance_km(indicators, drivable_roads, "dist_any_road_km")
indicators["no_road_2km"] = indicators["dist_any_road_km"] > R_NEAR_ROAD_KM
indicators["no_road_5km"] = indicators["dist_any_road_km"] > R_FAR_ROAD_KM


# %%
# =========================================================
# 8. SUPPLY: DISTANCE TO NEAREST TRANSMISSION LINE (local shapefile)
# =========================================================

transmission_shp = glob_first(TRANSMISSION_DIR, "*.shp")
transmission = gpd.read_file(transmission_shp)
if transmission.crs is None:
    transmission = transmission.set_crs(CRS_WGS84)
else:
    transmission = transmission.to_crs(CRS_WGS84)

indicators = add_nearest_distance_km(indicators, transmission, "dist_transmission_km")


# %%
# =========================================================
# 9. DEMAND: POPULATION (WorldPop pop_2km, pop_10km)
# =========================================================

worldpop_2030_path = os.path.join(CACHE_DIR, "worldpop_2030_constrained_100m.tif")
worldpop_fallback_path = os.path.join(CACHE_DIR, "worldpop_2020_1km_unadj.tif")

print("Fetching WorldPop population raster...")
if cached_download(WORLDPOP_2030_URL, worldpop_2030_path):
    worldpop_path = worldpop_2030_path
    pop_source_label = "worldpop_2030_constrained_100m_R2024B"
else:
    print("  2030 constrained product unavailable, falling back to 2020 1km UNadj.")
    cached_download(WORLDPOP_FALLBACK_URL, worldpop_fallback_path)
    worldpop_path = worldpop_fallback_path
    pop_source_label = "worldpop_2020_1km_UNadj_fallback"

try:
    indicators["pop_2km"] = zonal_stat_over_buffers(buffer_2km_wgs84, worldpop_path, stat="sum")
    indicators["pop_10km"] = zonal_stat_over_buffers(buffer_10km_wgs84, worldpop_path, stat="sum")
    indicators["pop_source"] = pop_source_label
except Exception as exc:
    print(f"  FAILED WorldPop zonal stats: {exc}")
    indicators["pop_2km"] = np.nan
    indicators["pop_10km"] = np.nan
    indicators["pop_source"] = np.nan


# %%
# =========================================================
# 10. DEMAND: BUILT-UP SURFACE (GHSL BUILT-S, builtup_m2_2km)
# =========================================================

ghsl_zip_path = os.path.join(CACHE_DIR, "GHS_BUILT_S_E2030_GLOBE_R2023A_54009_100_V1_0.zip")
ghsl_clip_path = os.path.join(CACHE_DIR, "ghsl_builtup_liberia_clip.tif")

ghsl_ok = os.path.exists(ghsl_clip_path)
if ghsl_ok:
    print(f"Using cached clipped GHSL raster: {ghsl_clip_path}")
else:
    ghsl_ok = cached_download(GHSL_ZIP_URL, ghsl_zip_path, timeout=600)
    if ghsl_ok:
        with zipfile.ZipFile(ghsl_zip_path) as zf:
            tif_members = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_members:
                raise FileNotFoundError("No .tif found inside GHSL zip")
            tif_member = tif_members[0]
        print(f"  clipping {tif_member} to Liberia bbox (windowed read, no full extraction)...")
        vsi_path = f"zip://{ghsl_zip_path}!{tif_member}"
        with rasterio.open(vsi_path) as src:
            bbox_ghsl_crs = gpd.GeoSeries([box(*liberia_bbox)], crs=CRS_WGS84).to_crs(src.crs).total_bounds
            window = rasterio.windows.from_bounds(*bbox_ghsl_crs, transform=src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(height=data.shape[0], width=data.shape[1], transform=transform)
            with rasterio.open(ghsl_clip_path, "w", **profile) as dst:
                dst.write(data, 1)
        print(f"  cached clipped GHSL raster: {ghsl_clip_path}")

try:
    if not ghsl_ok:
        raise FileNotFoundError("GHSL raster unavailable")
    indicators["builtup_m2_2km"] = zonal_stat_over_buffers(buffer_2km_wgs84, ghsl_clip_path, stat="sum")
except Exception as exc:
    print(f"  FAILED GHSL zonal stats: {exc}")
    indicators["builtup_m2_2km"] = np.nan


# %%
# =========================================================
# 11. SUPPLY: FLOOD HAZARD (JRC Global Flood Map, RP100y, flood_rp100)
# =========================================================

flood_zip_path = os.path.join(CACHE_DIR, "floodMapGL_rp100y.zip")
flood_clip_path = os.path.join(CACHE_DIR, "flood_rp100_liberia_clip.tif")

flood_ok = os.path.exists(flood_clip_path)
if flood_ok:
    print(f"Using cached clipped flood raster: {flood_clip_path}")
else:
    flood_ok = cached_download(JRC_FLOOD_ZIP_URL, flood_zip_path, timeout=600)
    if flood_ok:
        with zipfile.ZipFile(flood_zip_path) as zf:
            tif_members = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_members:
                raise FileNotFoundError("No .tif found inside JRC flood zip")
            tif_member = tif_members[0]
        vsi_path = f"zip://{flood_zip_path}!{tif_member}"
        with rasterio.open(vsi_path) as src:
            bbox_flood_crs = gpd.GeoSeries([box(*liberia_bbox)], crs=CRS_WGS84).to_crs(src.crs).total_bounds
            window = rasterio.windows.from_bounds(*bbox_flood_crs, transform=src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(height=data.shape[0], width=data.shape[1], transform=transform)
            with rasterio.open(flood_clip_path, "w", **profile) as dst:
                dst.write(data, 1)
        print(f"  cached clipped flood raster: {flood_clip_path}")

try:
    if not flood_ok:
        raise FileNotFoundError("Flood raster unavailable")
    indicators["flood_rp100"] = zonal_stat_over_buffers(buffer_1km_wgs84, flood_clip_path, stat="max")
except Exception as exc:
    print(f"  FAILED flood zonal stats: {exc}")
    indicators["flood_rp100"] = np.nan


# %%
# =========================================================
# 12. SUPPLY: LANDSLIDE HAZARD (GFDRR rainfall-triggered, landslide_precip)
# =========================================================

landslide_clip_path = os.path.join(CACHE_DIR, "landslide_precip_liberia_clip.tif")
landslide_fallback_full_path = os.path.join(CACHE_DIR, "LS_RF_Mean_1980-2018_COG.tif")


def _clip_raster_to_liberia(src, dst_path):
    bbox_crs = gpd.GeoSeries([box(*liberia_bbox)], crs=CRS_WGS84).to_crs(src.crs).total_bounds
    window = rasterio.windows.from_bounds(*bbox_crs, transform=src.transform)
    window = window.round_offsets().round_lengths()
    data = src.read(1, window=window)
    transform = src.window_transform(window)
    profile = src.profile.copy()
    profile.update(height=data.shape[0], width=data.shape[1], transform=transform)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)


landslide_ok = os.path.exists(landslide_clip_path)
if landslide_ok:
    print(f"Using cached clipped landslide raster: {landslide_clip_path}")
else:
    # Prefer a direct windowed read over the network (no full download) via
    # GDAL's /vsicurl/ support in rasterio. On a corporate network this can
    # fail even though `requests` (via truststore) succeeds, because GDAL's
    # bundled libcurl doesn't consult the Windows certificate store the way
    # truststore patches Python's ssl module to. If so, fall back to a plain
    # `requests` download of the full COG and clip locally instead.
    try:
        with rasterio.Env(GDAL_HTTP_UNSAFESSL="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(GFDRR_LANDSLIDE_COG_URL) as src:
                print("  reading landslide COG directly over the network (windowed)...")
                _clip_raster_to_liberia(src, landslide_clip_path)
        landslide_ok = True
        print(f"  cached clipped landslide raster: {landslide_clip_path}")
    except Exception as exc:
        print(f"  direct windowed COG read failed ({exc}); falling back to full requests download.")
        if cached_download(GFDRR_LANDSLIDE_COG_URL, landslide_fallback_full_path, timeout=900):
            with rasterio.open(landslide_fallback_full_path) as src:
                _clip_raster_to_liberia(src, landslide_clip_path)
            landslide_ok = True
            print(f"  cached clipped landslide raster: {landslide_clip_path}")

try:
    if not landslide_ok:
        raise FileNotFoundError("Landslide raster unavailable")
    indicators["landslide_precip"] = zonal_stat_over_buffers(buffer_1km_wgs84, landslide_clip_path, stat="max")
except Exception as exc:
    print(f"  FAILED landslide zonal stats: {exc}")
    indicators["landslide_precip"] = np.nan


# %%
# =========================================================
# 13. COMPLETENESS REPORT (increment 2: + raster indicators)
# =========================================================

increment2_cols = [
    "dist_hub_km", "dist_main_road_km", "dist_any_road_km",
    "no_road_2km", "no_road_5km", "dist_transmission_km",
    "pop_2km", "pop_10km", "builtup_m2_2km", "flood_rp100", "landslide_precip",
]
_ = completeness_report(indicators, increment2_cols)

print("\nIncrement 2 (+ raster indicators) complete.")
print(indicators[increment2_cols].describe(include="all"))


# %%
# =========================================================
# 14. DEMAND: BUNDLING / ANCHOR DENSITY (n_other_institutions_10km)
# =========================================================

buffer_10km_gdf = gpd.GeoDataFrame(
    {"institution_id": indicators["institution_id"].values},
    geometry=buffer_10km_m.values, crs=CRS_METRIC,
)
other_points_gdf = institutions_m[["institution_id", "geometry"]].rename(
    columns={"institution_id": "other_institution_id"}
)
joined_bundle = gpd.sjoin(buffer_10km_gdf, other_points_gdf, predicate="intersects", how="left")
bundle_counts = joined_bundle.groupby("institution_id")["other_institution_id"].count() - 1  # exclude self
indicators["n_other_institutions_10km"] = (
    indicators["institution_id"].map(bundle_counts).fillna(0).astype(int)
)


# %%
# =========================================================
# 15. DEMAND: BUILDINGS (n_buildings_2km, mean_building_area_2km)
# =========================================================

if BUILDING_SOURCE == "wsf":
    raise NotImplementedError(
        "BUILDING_SOURCE='wsf' is not implemented. Set BUILDING_SOURCE='open_buildings' "
        "(Google Open Buildings v3) or implement a WSF-based building count/area extractor."
    )
elif BUILDING_SOURCE != "open_buildings":
    raise ValueError(f"Unknown BUILDING_SOURCE: {BUILDING_SOURCE!r}")


def _s2_tokens_covering_bbox(bbox, level):
    minx, miny, maxx, maxy = bbox
    region = s2sphere.LatLngRect(
        s2sphere.LatLng.from_degrees(miny, minx),
        s2sphere.LatLng.from_degrees(maxy, maxx),
    )
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = 500
    return [c.to_token() for c in coverer.get_covering(region)]


ob_cache_dir = os.path.join(CACHE_DIR, "open_buildings")
os.makedirs(ob_cache_dir, exist_ok=True)

ob_tokens = _s2_tokens_covering_bbox(liberia_bbox, OPEN_BUILDINGS_S2_LEVEL)
print(f"Open Buildings: {len(ob_tokens)} S2 level-{OPEN_BUILDINGS_S2_LEVEL} tiles cover Liberia bbox.")

building_frames = []
for token in ob_tokens:
    tile_path = os.path.join(ob_cache_dir, f"{token}_buildings.csv.gz")
    tile_url = f"{OPEN_BUILDINGS_BASE_URL}/{token}_buildings.csv.gz"
    if not os.path.exists(tile_path):
        if not cached_download(tile_url, tile_path, timeout=300):
            continue  # tile has no buildings (e.g. ocean) or is unavailable
    try:
        tile_df = pd.read_csv(tile_path, compression="gzip", header=None, names=OPEN_BUILDINGS_COLUMNS)
        building_frames.append(tile_df[["latitude", "longitude", "area_in_meters"]])
    except Exception as exc:
        print(f"  skipping unreadable tile {tile_path}: {exc}")

try:
    if not building_frames:
        raise ValueError("No Open Buildings tiles could be read")
    buildings_df = pd.concat(building_frames, ignore_index=True)
    print(f"  loaded {len(buildings_df):,} building points across {len(ob_tokens)} tiles")
    buildings_gdf = gpd.GeoDataFrame(
        buildings_df,
        geometry=gpd.points_from_xy(buildings_df["longitude"], buildings_df["latitude"]),
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    buffer_2km_gdf = gpd.GeoDataFrame(
        {"institution_id": indicators["institution_id"].values},
        geometry=buffer_2km_m.values, crs=CRS_METRIC,
    )
    joined_bld = gpd.sjoin(
        buffer_2km_gdf, buildings_gdf[["area_in_meters", "geometry"]],
        predicate="intersects", how="left",
    )
    bld_agg = joined_bld.groupby("institution_id")["area_in_meters"].agg(["count", "mean"])
    indicators["n_buildings_2km"] = indicators["institution_id"].map(bld_agg["count"]).fillna(0).astype(int)
    indicators["mean_building_area_2km"] = indicators["institution_id"].map(bld_agg["mean"])
except Exception as exc:
    print(f"  FAILED Open Buildings extraction: {exc}")
    indicators["n_buildings_2km"] = np.nan
    indicators["mean_building_area_2km"] = np.nan


# %%
# =========================================================
# 16. DEMAND: BUILDING HEIGHT (WSF3D, mean_building_height_2km)
# =========================================================

if WSF3D_BH_URL is None:
    print("WSF3D_BH_URL not configured - skipping building height (filling NaN).")
    indicators["mean_building_height_2km"] = np.nan
else:
    wsf3d_path = os.path.join(CACHE_DIR, "wsf3d_building_height.tif")
    wsf3d_clip_path = os.path.join(CACHE_DIR, "wsf3d_building_height_liberia_clip.tif")
    try:
        if not os.path.exists(wsf3d_clip_path):
            if not cached_download(WSF3D_BH_URL, wsf3d_path, timeout=600):
                raise FileNotFoundError("WSF3D download failed")
            with rasterio.open(wsf3d_path) as src:
                _clip_raster_to_liberia(src, wsf3d_clip_path)
        indicators["mean_building_height_2km"] = zonal_stat_over_buffers(
            buffer_2km_wgs84, wsf3d_clip_path, stat="mean"
        )
    except Exception as exc:
        if SKIP_IF_UNAVAILABLE:
            print(f"  WSF3D unavailable ({exc}); filling NaN (SKIP_IF_UNAVAILABLE=True).")
            indicators["mean_building_height_2km"] = np.nan
        else:
            raise


# %%
# =========================================================
# 17. DEMAND: RELATIVE WEALTH INDEX (rwi_popweighted_10km, rwi_method)
# =========================================================

if not os.path.exists(RWI_CSV_PATH):
    print(
        f"RWI_CSV_PATH not found: {RWI_CSV_PATH}\n"
        "  Download Liberia's Relative Wealth Index CSV (lat/lon/rwi grid, ~2.4km "
        "spacing) manually from the Data for Good at Meta / HDX RWI page "
        "(https://data.humdata.org, search 'Relative Wealth Index') and place it "
        "at this path, then re-run this cell."
    )
    indicators["rwi_popweighted_10km"] = np.nan
    indicators["rwi_method"] = "rwi_csv_missing"
else:
    rwi_raw = pd.read_csv(RWI_CSV_PATH)
    col_map = {c.lower(): c for c in rwi_raw.columns}
    lat_col = col_map.get("latitude", col_map.get("lat"))
    lon_col = col_map.get("longitude", col_map.get("lon", col_map.get("long")))
    rwi_col = col_map.get("rwi")
    if not (lat_col and lon_col and rwi_col):
        raise ValueError(f"Could not find lat/lon/rwi columns in {RWI_CSV_PATH}: {rwi_raw.columns.tolist()}")

    rwi_gdf = gpd.GeoDataFrame(
        rwi_raw, geometry=gpd.points_from_xy(rwi_raw[lon_col], rwi_raw[lat_col]), crs=CRS_WGS84
    ).to_crs(CRS_METRIC)

    try:
        with rasterio.open(worldpop_path) as src:
            coords = [(geom.x, geom.y) for geom in rwi_gdf.to_crs(src.crs).geometry]
            rwi_gdf["pop_weight"] = [v[0] if v[0] is not None else 0 for v in src.sample(coords)]
    except Exception as exc:
        print(f"  could not sample WorldPop at RWI points ({exc}); using equal weights.")
        rwi_gdf["pop_weight"] = 1.0
    rwi_gdf["pop_weight"] = rwi_gdf["pop_weight"].clip(lower=0)

    buffer_10km_gdf_rwi = gpd.GeoDataFrame(
        {"institution_id": indicators["institution_id"].values},
        geometry=buffer_10km_m.values, crs=CRS_METRIC,
    )
    joined_rwi = gpd.sjoin(
        buffer_10km_gdf_rwi, rwi_gdf[[rwi_col, "pop_weight", "geometry"]],
        predicate="intersects", how="left",
    )

    def _weighted_mean(group):
        vals = group[rwi_col]
        mask = vals.notna()
        if mask.sum() == 0:
            return np.nan
        w = group.loc[mask, "pop_weight"]
        if w.sum() <= 0:
            return vals[mask].mean()
        return np.average(vals[mask], weights=w)

    rwi_grouped = joined_rwi.groupby("institution_id").apply(_weighted_mean, include_groups=False)
    indicators["rwi_popweighted_10km"] = indicators["institution_id"].map(rwi_grouped)
    indicators["rwi_method"] = np.where(indicators["rwi_popweighted_10km"].notna(), "popweighted_10km", "")

    # Fallback: nearest RWI point for institutions with none within 10km.
    missing_mask = indicators["rwi_popweighted_10km"].isna()
    if missing_mask.any():
        inst_missing_m = institutions_m.loc[missing_mask.values, ["institution_id", "geometry"]]
        nearest_rwi = gpd.sjoin_nearest(
            inst_missing_m, rwi_gdf[[rwi_col, "geometry"]], how="left", distance_col="dist_m"
        )
        nearest_rwi = nearest_rwi.groupby("institution_id").first()
        indicators.loc[missing_mask, "rwi_popweighted_10km"] = (
            indicators.loc[missing_mask, "institution_id"].map(nearest_rwi[rwi_col])
        )
        indicators.loc[missing_mask, "rwi_method"] = "nearest_point_fallback"

    print(f"  RWI method counts:\n{indicators['rwi_method'].value_counts()}")


# %%
# =========================================================
# 18. OPTIONAL FALLBACK + CROSS-CHECK ONLY: DRE ATLAS
#     (security_risk_fallback, dreatlas_pop_nearest, dreatlas_demand_nearest)
#     Never used in MVI/PCI scoring except security_risk_fallback when
#     USE_SECURITY=True; dreatlas_* columns are cross-check only.
# =========================================================

dre_raw = pd.read_csv(DRE_ATLAS_CSV)
dre_gdf = gpd.GeoDataFrame(dre_raw, geometry=dre_raw["geometry"].apply(_wkt.loads), crs=CRS_WGS84)
dre_gdf_m = dre_gdf.to_crs(CRS_METRIC)
dre_points_m = dre_gdf_m.copy()
dre_points_m["geometry"] = dre_points_m.geometry.representative_point()

nearest_dre = gpd.sjoin_nearest(
    institutions_m[["institution_id", "geometry"]],
    dre_points_m[["population", "demand", "security_risk", "geometry"]],
    how="left", distance_col="dist_m",
)
nearest_dre = nearest_dre.groupby("institution_id").first()

indicators["dreatlas_pop_nearest"] = indicators["institution_id"].map(nearest_dre["population"])
indicators["dreatlas_demand_nearest"] = indicators["institution_id"].map(nearest_dre["demand"])
indicators["dreatlas_match_km"] = indicators["institution_id"].map(nearest_dre["dist_m"]) / 1000

if USE_SECURITY:
    indicators["security_risk_fallback"] = indicators["institution_id"].map(nearest_dre["security_risk"])
    indicators["security_match_km"] = indicators["dreatlas_match_km"]
    print("USE_SECURITY=True: security_risk_fallback populated from nearest DRE Atlas settlement (fallback field).")
else:
    indicators["security_risk_fallback"] = np.nan
    indicators["security_match_km"] = np.nan
    print("USE_SECURITY=False: security_risk_fallback left as NaN (DRE Atlas settlement fallback disabled).")


# %%
# =========================================================
# 19. COMPLETENESS REPORT (increment 3: full indicator set)
# =========================================================

all_score_cols = [
    "pop_2km", "pop_10km", "pop_source", "builtup_m2_2km",
    "n_buildings_2km", "mean_building_area_2km", "mean_building_height_2km",
    "rwi_popweighted_10km", "rwi_method", "n_other_institutions_10km",
    "dist_hub_km", "dist_main_road_km", "dist_any_road_km", "no_road_2km", "no_road_5km",
    "dist_transmission_km", "flood_rp100", "landslide_precip",
    "security_risk_fallback", "security_match_km",
    "dreatlas_pop_nearest", "dreatlas_demand_nearest", "dreatlas_match_km",
]
completeness_df = completeness_report(indicators, all_score_cols)
print("\nIncrement 3 (full indicator set) complete.")


# %%
# =========================================================
# 20. SAVE OUTPUTS
# =========================================================

out_gpkg = os.path.join(OUTPUT_DIR, "institution_indicators.gpkg")
indicators.to_file(out_gpkg, layer="indicators", driver="GPKG")

indicators_csv = indicators.copy()
indicators_csv["longitude"] = indicators_csv.geometry.x
indicators_csv["latitude"] = indicators_csv.geometry.y
indicators_csv = pd.DataFrame(indicators_csv.drop(columns="geometry"))
out_csv = os.path.join(OUTPUT_DIR, "institution_indicators.csv")
indicators_csv.to_csv(out_csv, index=False)

completeness_path = os.path.join(OUTPUT_DIR, "institution_indicators_completeness.csv")
completeness_df.to_csv(completeness_path, index=False)

print("Saved:")
print(out_gpkg)
print(out_csv)
print(completeness_path)
