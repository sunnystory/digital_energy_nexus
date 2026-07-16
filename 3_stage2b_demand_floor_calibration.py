# -*- coding: utf-8 -*-
"""
Stage 2b — Demand-Floor Calibration (Liberia)

Purpose
-------
Derives the two "demand floor" values used by Stage 2's new-build gates from
Liberia's OWN existing infrastructure, instead of hand-picking numbers:

    MIN_POP_MINIGRID_2KM   minimum near-field population (2 km) below which a
                           NEW standalone mini-grid (rung E4) is not proposed.
    MIN_POP_TOWER_10KM     minimum catchment population (10 km) below which a
                           NEW tower (rung C3) is not proposed.

Method: revealed-deployment calibration
---------------------------------------
Every EXISTING NEA mini-grid site and every EXISTING telecom tower already
represents a community where someone judged the market big enough to build.
This script measures, for each existing site, the same population indicator
that Stage 2a computes for institutions:

    mini-grid sites -> population within 2 km  (the walk-in service area of a
                       mini-grid's low-voltage distribution network)
    tower sites     -> population within 10 km (roughly a rural macro-cell's
                       coverage catchment)

The LOWER TAIL of each distribution (e.g. the 10th percentile, p10) is then
"the smallest community Liberia has actually built this infrastructure for",
and becomes the screening floor. Because the exact same buffer + WorldPop
routine is used for the reference sites and for the institutions being
screened, any systematic bias in the indicator (buffer dilution, raster
error) enters both sides equally and cancels out of the comparison.

External cross-checks (reported alongside, never used as the floor itself):
  - ESMAP, "Mini Grids for Half a Billion People" (World Bank, 2022):
    installed mini-grids worldwide serve very roughly 1,000-2,000 people per
    system on average, and the planning pipeline trends toward ~1,200.
  - GSMA Connected Society analysis of Liberia: 94% of settlements above
    ~4,000 people are already covered; the uncovered population lives in
    settlements averaging ~80 people. The commercial frontier for an
    unsubsidised tower therefore sits in the thousands; an anchor-supported
    (ABC+) tower can go somewhat lower, but nowhere near 80.

Run order:  Stage 1 -> Stage 2a -> THIS SCRIPT -> Stage 2.
(Fast if Stage 2a already cached the WorldPop raster.)

Outputs (in OUTPUT_DIR):
  demand_floor_calibration.csv          percentile table Stage 2 auto-reads
  calibration_minigrid_sites.csv        per-site pop_2km for every mini-grid
  calibration_tower_sites.csv           per-site pop_10km for every tower

Run as a script, or cell-by-cell using the "# %%" markers.
"""

# %%
import truststore
truststore.inject_into_ssl()

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterstats


# =========================================================
# 1. CONFIGURATION  (paths mirror Stage 1 / Stage 2a)
# =========================================================

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32629"  # UTM Zone 29N

DOWNLOADS_DIR = r"C:\Users\wb632724\Downloads"
BASE_DIR = os.path.join(DOWNLOADS_DIR, "e&d")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "delivery_matching")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Existing infrastructure (identical to Stage 1's inputs)
ENERGY_DIR = os.path.join(DOWNLOADS_DIR, "energy_data_liberia")
MINIGRID_PATH = os.path.join(ENERGY_DIR, "liberia-small-mini-grid-nea-project", "nea_small_minigrid.shp")
TOWER_CSV_PATH = os.path.join(DOWNLOADS_DIR, "all_matched_sites_sharada (1).csv")
TOWER_LON_COL = "Longitude"
TOWER_LAT_COL = "Latitude"
TOWER_TECH_COL = "Technology"

# WorldPop raster (identical to Stage 2a; reuses its cache if present)
WORLDPOP_2030_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2030/"
    "LBR/v1/100m/constrained/lbr_pop_2030_CN_100m_R2024B_v1.tif"
)
WORLDPOP_FALLBACK_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/LBR/"
    "lbr_ppp_2020_UNadj.tif"
)

# Buffer radii: MUST equal Stage 2a's institution buffers so the floor and
# the screened indicator are the same quantity.
R_MINIGRID_SERVICE_KM = 2    # = Stage 2a R_COMMUNITY_KM (pop_2km)
R_TOWER_CATCHMENT_KM = 10    # = Stage 2a R_CATCHMENT_KM (pop_10km)

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

# External reference points, reported for cross-checking only.
ESMAP_2022_TYPICAL_POP_PER_MINIGRID = "~1,000-2,000 (installed avg ~2,200; pipeline avg ~1,200)"
GSMA_LIBERIA_COMMERCIAL_TOWER_FRONTIER = "settlements >= ~4,000 people 94% covered; uncovered avg ~80"

CALIBRATION_CSV = os.path.join(OUTPUT_DIR, "demand_floor_calibration.csv")


# =========================================================
# 2. HELPERS (same conventions as Stage 2a)
# =========================================================

def cached_download(url, dest_path, timeout=600, chunk_size=1 << 20):
    import requests
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
        print(f"  saved: {dest_path}")
        return True
    except Exception as exc:
        print(f"  FAILED download {url}: {exc}")
        if os.path.exists(dest_path + ".part"):
            os.remove(dest_path + ".part")
        return False


def zonal_pop_sum(points_gdf_wgs84, radius_km, raster_path):
    """
    Population within radius_km of each point: buffer in the metric CRS,
    reproject the buffers into the raster's own CRS (never resample the
    raster), zonal SUM with all_touched=True. Identical logic to Stage 2a's
    pop_2km / pop_10km so the calibrated floor and the screened indicator
    are directly comparable.
    """
    pts_m = points_gdf_wgs84.to_crs(CRS_METRIC)
    buffers_m = pts_m.geometry.buffer(radius_km * 1000)
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
    buffers_raster = gpd.GeoSeries(buffers_m, crs=CRS_METRIC).to_crs(raster_crs)
    results = rasterstats.zonal_stats(
        buffers_raster.geometry, raster_path, stats=["sum"],
        all_touched=True, nodata=nodata,
    )
    return np.array([r["sum"] if r["sum"] is not None else np.nan for r in results])


def percentile_row(values, label, indicator):
    values = pd.Series(values).dropna()
    row = {"infrastructure": label, "indicator": indicator, "n_sites": len(values),
           "mean": values.mean()}
    for p in PERCENTILES:
        row[f"p{p:02d}"] = values.quantile(p / 100)
    print(f"\n--- {label} ({indicator}), n={len(values)} ---")
    print(pd.Series({f"p{p:02d}": row[f"p{p:02d}"] for p in PERCENTILES}).round(0))
    return row


# %%
# =========================================================
# 3. LOAD EXISTING INFRASTRUCTURE SITES
# =========================================================

print("Loading existing mini-grid sites...")
minigrid = gpd.read_file(MINIGRID_PATH)
if minigrid.crs is None:
    minigrid = minigrid.set_crs(CRS_WGS84)
else:
    minigrid = minigrid.to_crs(CRS_WGS84)
# collapse any polygons/lines to representative points, as Stage 1 does
minigrid_m = minigrid.to_crs(CRS_METRIC)
minigrid_m["geometry"] = minigrid_m.geometry.representative_point()
minigrid_pts = minigrid_m.to_crs(CRS_WGS84).reset_index(drop=True)
print(f"  {len(minigrid_pts):,} mini-grid sites")

print("Loading existing tower sites...")
tower_df = pd.read_csv(TOWER_CSV_PATH)
tower_df = tower_df.dropna(subset=[TOWER_LON_COL, TOWER_LAT_COL])
towers = gpd.GeoDataFrame(
    tower_df,
    geometry=gpd.points_from_xy(tower_df[TOWER_LON_COL], tower_df[TOWER_LAT_COL]),
    crs=CRS_WGS84,
).reset_index(drop=True)
# keep only sites whose technology string parses (same rule as Stage 1)
tech_upper = towers[TOWER_TECH_COL].astype(str).str.upper()
towers = towers[tech_upper.str.contains("2G|3G|4G", regex=True)].reset_index(drop=True)
print(f"  {len(towers):,} tower sites with a parseable technology string")


# %%
# =========================================================
# 4. POPULATION RASTER (reuses Stage 2a's cache)
# =========================================================

worldpop_2030_path = os.path.join(CACHE_DIR, "worldpop_2030_constrained_100m.tif")
worldpop_fallback_path = os.path.join(CACHE_DIR, "worldpop_2020_1km_unadj.tif")

if cached_download(WORLDPOP_2030_URL, worldpop_2030_path):
    worldpop_path = worldpop_2030_path
    pop_source_label = "worldpop_2030_constrained_100m_R2024B"
else:
    print("  2030 constrained product unavailable, falling back to 2020 1km UNadj.")
    cached_download(WORLDPOP_FALLBACK_URL, worldpop_fallback_path)
    worldpop_path = worldpop_fallback_path
    pop_source_label = "worldpop_2020_1km_UNadj_fallback"
print(f"Population raster: {pop_source_label}")


# %%
# =========================================================
# 5. MEASURE POPULATION AROUND EACH EXISTING SITE
# =========================================================

print("\nComputing pop_2km around each existing mini-grid site...")
minigrid_pts["pop_2km"] = zonal_pop_sum(minigrid_pts, R_MINIGRID_SERVICE_KM, worldpop_path)

print("Computing pop_10km around each existing tower site...")
towers["pop_10km"] = zonal_pop_sum(towers, R_TOWER_CATCHMENT_KM, worldpop_path)

rows = [
    percentile_row(minigrid_pts["pop_2km"], "minigrid", "pop_2km"),
    percentile_row(towers["pop_10km"], "tower", "pop_10km"),
]

summary = pd.DataFrame(rows)
summary["pop_source"] = pop_source_label
summary["reference_note"] = [
    f"ESMAP 2022 typical population per mini-grid: {ESMAP_2022_TYPICAL_POP_PER_MINIGRID}",
    f"GSMA Liberia coverage analysis: {GSMA_LIBERIA_COMMERCIAL_TOWER_FRONTIER}",
]


# %%
# =========================================================
# 6. SAVE + INTERPRETATION GUIDANCE
# =========================================================

summary.to_csv(CALIBRATION_CSV, index=False)
minigrid_out = os.path.join(OUTPUT_DIR, "calibration_minigrid_sites.csv")
tower_out = os.path.join(OUTPUT_DIR, "calibration_tower_sites.csv")

mg_csv = minigrid_pts.copy()
mg_csv["longitude"] = mg_csv.geometry.x
mg_csv["latitude"] = mg_csv.geometry.y
pd.DataFrame(mg_csv.drop(columns="geometry")).to_csv(minigrid_out, index=False)

tw_csv = towers.copy()
tw_csv["longitude"] = tw_csv.geometry.x
tw_csv["latitude"] = tw_csv.geometry.y
pd.DataFrame(tw_csv.drop(columns="geometry")).to_csv(tower_out, index=False)

print("\nSaved:")
print(f"  {CALIBRATION_CSV}   <- Stage 2 reads this automatically")
print(f"  {minigrid_out}")
print(f"  {tower_out}")

print(f"""
============================================================
HOW TO READ THIS (before running Stage 2)
============================================================
1. Look at the p10 / p25 columns above.
   - minigrid pop_2km p10 : sanity range is the high hundreds to low
     thousands. ESMAP's global reference is {ESMAP_2022_TYPICAL_POP_PER_MINIGRID}.
   - tower pop_10km p10   : expect thousands to tens of thousands (a 10 km
     catchment aggregates several settlements, so it should sit well above
     the ~4,000-person single-settlement commercial frontier from GSMA's
     Liberia analysis).
2. If a percentile looks implausible (e.g. p10 near zero), open the per-site
   CSV, sort ascending, and inspect the low-tail sites: coordinate errors
   and urban infill sites are the usual culprits. Remove/flag and re-run.
3. Stage 2 will, by default, read the p10 row of this CSV as the floor
   (FLOOR_PERCENTILE = 10). If the +/-50%% sensitivity report in Stage 2
   shows large swings in E4/C3 counts, move to p25 (more conservative) or
   agree a value with the Energy/Digital teams.
============================================================
""")
