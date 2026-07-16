# -*- coding: utf-8 -*-
"""
Stage 2 — Delivery Matching (Liberia), Rev. 3: demand-floor gates

Reads institution_indicators.gpkg (Stage 2a's output; institution-level
buffer/distance indicators + every Stage 1 column carried forward) and scores
each institution with no settlement join anywhere in the scoring path.

DESIGN (three tiers, replacing the Rev. 2 median-split segmentation):
  Tier 1 - Physical decision rules (deterministic, distance-based):
           E0-E3, C0-C2, plus C3's road-access condition. Unchanged in
           substance; C-ladder evaluation order fixed so cheaper options
           (C2 extension) are tested before more expensive ones (C3 new
           tower), matching the "cheapest credible option first" principle.
  Tier 2 - Demand floors (absolute, evidence-based) for the two rungs that
           propose NEW revenue-dependent infrastructure:
             E4 (new mini-grid): pop_2km  >= MIN_POP_MINIGRID_2KM
             C3 (new tower):     pop_10km >= MIN_POP_TOWER_10KM (+ road <=5km)
           Floors are framed as EXCLUSION conditions ("below this, proposing
           new revenue-dependent infrastructure is not credible"), not
           success guarantees. They are resolved in this order:
             (1) explicit CONFIG value, if set;
             (2) the calibration CSV written by stage2b_demand_floor_
                 calibration.py (revealed-deployment percentile,
                 FLOOR_PERCENTILE, default p10);
             (3) literature-anchored placeholder defaults (ESMAP 2022 /
                 GSMA Liberia), with a loud warning.
  Tier 3 - Continuous priority ranking (unchanged priority_score weights).
           Rung-internal ordering = filter the global ranking by rung; no
           separate score.

M1-M4 MARKET SEGMENTS ARE REMOVED ENTIRELY (Rev. 3). The Rev. 2 median-split
gate (new-build requires MVI >= the global median) filtered out 100% of gap
sites in the Liberia run, because gap sites are disproportionately remote
and low-density by construction while the median is computed over ALL
institutions including urban ones. The demand floors above replace that gate
with an absolute, literature/calibration-anchored test. MVI and PCI remain
as continuous inputs to the priority score only.

ROAD-ACCESS CONDITION (C3/C4 vs C5): the 5 km "buildability" screen applies
to the connectivity ladder only (kept asymmetric by team decision - the
energy fallback E5 is a man-portable standalone solar system, so the E
ladder needs no road branch). Provenance of the 5 km value: the strict
international standard for personal rural road access is 2 km to an
all-season road (World Bank Rural Access Index, SDG indicator 9.1.1, a
20-25 minute walk); construction/O&M logistics tolerate more than daily
walking access, so the buildability screen sits one band out at 5 km,
matching (a) the most favorable distance-to-roads class (0-5 km) in the
OnSSET geospatial electrification suitability framework (Mentis et al.
2017, ERL 12, appendix) and (b) Stage 2a's own no_road_5km "severe
isolation" flag, which this script reuses.

Computes, per institution:
    MVI / PCI                             continuous composites; priority
                                          score inputs only (no gating role).
    energy_ladder (E0-E5),
    connectivity_ladder (C0-C5)           extension-feasibility ladders.
    meets_minigrid_demand_floor,
    meets_tower_demand_floor              Tier 2 flags (audit trail).
    joint_delivery_type / delivery_model  keyed off Stage 1's gap status.
    type_b_window_flag                    demand-side overlay (toolkit Type B:
                                          household/MSME solar-kit + device
                                          affordability windows) - NOT an
                                          alternative to the site options.
    priority_score / priority_rank        see PRIORITY_WEIGHTS.
    lot_cluster_id / procurement_lot_id   geographic procurement lots.

Diagnostics printed at the end:
    - floor sensitivity report (x0.5 .. x1.5): E4 / C3 counts under scaled
      floors, so threshold-boundary fragility is visible before decisions;
    - DRE Atlas cross-check (median dreatlas_pop_nearest per rung; NEVER a
      score input) to verify the floors separate rungs on independent data.

Run order: Stage 1 -> Stage 2a -> Stage 2b (calibration) -> this script.
Run as a script, or cell-by-cell in VSCode / Jupyter using the "# %%" markers.
"""

# %%
import truststore
truststore.inject_into_ssl()

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import unary_union

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        pass


# =========================================================
# 1. CONFIGURATION
# =========================================================

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32629"  # UTM Zone 29N

BASE_DIR = r"C:\Users\wb632724\Downloads\e&d"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "delivery_matching")

INDICATORS_GPKG = os.path.join(OUTPUT_DIR, "institution_indicators.gpkg")
INDICATORS_LAYER = "indicators"

# --- MVI (Market Viability Index) weights; must sum to 1 ---
# transform: "log" = log1p before percentile normalization.
MVI_WEIGHTS = {
    "pop_2km":                  {"weight": 0.20, "log": True},
    "pop_10km":                 {"weight": 0.10, "log": True},
    "builtup_m2_2km":           {"weight": 0.10, "log": True},
    "n_buildings_2km":          {"weight": 0.15, "log": True},
    "mean_building_height_2km": {"weight": 0.10, "log": False},
    "rwi_popweighted_10km":     {"weight": 0.25, "log": False},
    "n_other_institutions_10km": {"weight": 0.10, "log": True},
}

# --- PCI (Physical Constraint Index) weights; must sum to 1 ---
# Hazard columns use fillna0_hazard=True because a NaN in the flood/landslide
# rasters means "no risk modeled at this location" (e.g. JRC's flood map only
# carries values in mapped river floodplains), not "unknown" - so NaN -> 0
# risk rather than being dropped/renormalized like a genuinely missing column.
PCI_CAP_HUB_KM = 50
PCI_CAP_MAIN_ROAD_KM = 10
PCI_CAP_TRANSMISSION_KM = 80
PCI_ROAD_BASE_WEIGHT = 0.10   # no_road_2km
PCI_ROAD_SEVERE_BONUS = 0.05  # extra if no_road_5km also true (severe isolation)

PCI_WEIGHTS = {
    "dist_hub_km":          {"weight": 0.20, "log": False, "cap": PCI_CAP_HUB_KM},
    "dist_main_road_km":    {"weight": 0.15, "log": False, "cap": PCI_CAP_MAIN_ROAD_KM},
    "road_isolation_norm":  {"weight": PCI_ROAD_BASE_WEIGHT, "log": False, "cap": None},
    "dist_transmission_km": {"weight": 0.20, "log": False, "cap": PCI_CAP_TRANSMISSION_KM},
    "flood_rp100":          {"weight": 0.20, "log": False, "cap": None, "fillna0_hazard": True},
    "landslide_precip":     {"weight": 0.15, "log": False, "cap": None, "fillna0_hazard": True},
}

# Optional security fallback (DRE Atlas nearest-settlement security_risk).
# Only takes effect if the column is actually populated (stage2a's
# USE_SECURITY=True); otherwise the generic weighting helper drops it as
# all-NaN and renormalizes automatically, same as any other missing column.
USE_SECURITY = False
SECURITY_WEIGHT = 0.15
SECURITY_RISK_ORDINAL_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}

PCT_LOW, PCT_HIGH = 0.05, 0.95  # percentile min-max normalization bounds

# --- Ladders: extension-feasibility bands for GAP sites only ---
# Stage 1's access proxy already answers "is this site served?" (line/
# transformer <=5km, mini-grid <=2km -> access proxy True). Reusing that same
# 5km/2km band for "can this gap site be reached by extending existing
# infrastructure?" collapsed almost every gap site into rung 0 (both are gaps
# by definition once you're past the proxy threshold), with no gradient
# between a site 6km from the grid and one 60km away. These bands start
# where the access-proxy band ends and run out to a screening cutoff instead.
ENERGY_ACCESS_LINE_KM = 5      # mirrors Stage 1's line/transformer access-proxy threshold
ENERGY_ACCESS_MINIGRID_KM = 2  # mirrors Stage 1's mini-grid access-proxy threshold
GRID_EXTENSION_MAX_KM = 15     # screening cutoff: MV-line extension vs. off-grid cost crossover (~5-15km); retune per LEC/NEA unit costs
MINIGRID_EXTENSION_MAX_KM = 10 # screening cutoff for mini-grid extension candidates
TOWER_EXTENSION_MAX_KM = 15    # screening cutoff: beyond typical rural macro-cell radius (8-10km) but still infill/sector-replan range

assert GRID_EXTENSION_MAX_KM > ENERGY_ACCESS_LINE_KM, "grid extension band must start after the access-proxy band"
assert MINIGRID_EXTENSION_MAX_KM > ENERGY_ACCESS_MINIGRID_KM, "mini-grid extension band must start after the access-proxy band"

# --- Tier 2: demand floors for NEW revenue-dependent infrastructure ---
# Exclusion screens: BELOW the floor, proposing a new mini-grid (E4) or a new
# tower (C3) is not credible; the site falls through to the service-model
# default (E5 / C4-C5). Floors are NOT success guarantees above the line.
#
# Resolution order (see resolve_demand_floor):
#   1. explicit value here (set a number to override everything);
#   2. calibration CSV from stage2b_demand_floor_calibration.py -
#      "revealed-deployment" percentile of pop_2km around EXISTING mini-grids
#      / pop_10km around EXISTING towers (FLOOR_PERCENTILE, default p10 =
#      the smallest communities Liberia has actually built for);
#   3. literature-anchored placeholders, with a warning:
#      - 1,000 for mini-grids: lower edge of ESMAP 2022's typical population
#        per system (installed avg ~2,200; planning pipeline avg ~1,200);
#      - 3,000 for towers: below GSMA's Liberia commercial frontier
#        (settlements >= ~4,000 people are 94% covered) to reflect the
#        subsidised/anchor-supported ABC+ model, while noting a 10 km
#        catchment aggregates several settlements (an upward pressure) -
#        i.e. a neutral starting point pending calibration.
MIN_POP_MINIGRID_2KM = None      # None -> resolve from calibration CSV, else placeholder
MIN_POP_TOWER_10KM = None        # None -> resolve from calibration CSV, else placeholder
FLOOR_PERCENTILE = 10            # which calibration percentile becomes the floor (10 or 25)
CALIBRATION_CSV = os.path.join(OUTPUT_DIR, "demand_floor_calibration.csv")
PLACEHOLDER_MIN_POP_MINIGRID_2KM = 1000   # ESMAP 2022-anchored default
PLACEHOLDER_MIN_POP_TOWER_10KM = 3000     # GSMA Liberia-anchored default
FLOOR_SENSITIVITY_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]

# --- Priority score weights; must sum to 1 ---
PRIORITY_WEIGHTS = {
    "beneficiary_scale": 0.35,   # from pop_2km (log, percentile-normalized)
    "access_gap": 0.35,          # from Stage 1 combined_access_status
    "market_viability": 0.20,    # MVI
    "constraint_ease": 0.10,     # 1 - PCI (easier/cheaper to deliver sooner)
}
ACCESS_GAP_SCORE = {
    "No energy or digital access": 1.0,
    "Energy only": 0.5,
    "Digital only": 0.5,
    "Energy + Digital access": 0.0,
}

# --- Type B demand-side overlay (toolkit Section 3.2: joint facility windows
# for household/MSME off-grid solar kits + digital devices) ---
# Rev. 3 rule: flag institutions whose location suggests the surrounding
# community is a candidate for an affordability window run IN PARALLEL with
# the site-level contract: any residual access gap at the site, OR a
# low-wealth catchment (population-weighted RWI at or below the threshold).
# The Rev. 2 flag also required segment M1-M3; with segments removed, that
# condition is gone - which also resolves the Rev. 2 self-contradiction where
# rwi pushed MVI up (segment condition) and the flag down (poverty condition)
# on the same variable, so the flag could never fire once real RWI loaded.
TYPE_B_RWI_THRESHOLD = -0.3   # Meta Relative Wealth Index units (0 = country mean)

# --- Procurement lotting ---
LOT_CLUSTER_RADIUS_KM = 15   # institutions within this radius bundle into one lot cluster
MAX_INSTITUTIONS_PER_LOT = 25


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def _percentile_normalize(values, pct_low=PCT_LOW, pct_high=PCT_HIGH):
    """Clip to [0,1] using the pct_low-pct_high percentile range as the scale."""
    lo, hi = values.quantile(pct_low), values.quantile(pct_high)
    if hi > lo:
        norm = (values - lo) / (hi - lo)
    else:
        norm = pd.Series(0.5, index=values.index)  # degenerate: no spread in the data
    return norm.clip(0, 1)


def build_weighted_index(df, weight_spec, index_label):
    """
    Generic MVI/PCI-style weighted index builder.

    weight_spec: {col: {"weight": float, "log": bool, "cap": float|None,
                         "fillna0_hazard": bool}}
    Drops any column that is entirely NaN (printing a warning identifying it),
    renormalizes the remaining weights to sum to 1, and returns:
        (index_series in [0,1], normalized_components_df, weights_actually_used)
    """
    nominal_total = sum(spec["weight"] for spec in weight_spec.values())
    assert abs(nominal_total - 1.0) < 1e-6, (
        f"{index_label} CONFIG weights must sum to 1, got {nominal_total:.4f}"
    )

    normalized = {}
    available_weights = {}
    for col, spec in weight_spec.items():
        if col not in df.columns or df[col].isna().all():
            print(f"  *** WARNING: '{col}' unavailable (all-NaN) for {index_label} - "
                  f"dropping its weight ({spec['weight']:.3f}) and renormalizing "
                  f"the remaining {index_label} weights. ***")
            continue
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        if spec.get("fillna0_hazard", False):
            values = values.fillna(0.0)
        if spec.get("log", False):
            values = np.log1p(values.clip(lower=0))
        if spec.get("cap") is not None:
            values = values.clip(upper=spec["cap"])
        norm = _percentile_normalize(values)
        norm = norm.fillna(norm.median())
        normalized[col] = norm
        available_weights[col] = spec["weight"]

    weight_sum = sum(available_weights.values())
    renorm_weights = {c: w / weight_sum for c, w in available_weights.items()}
    print(f"  {index_label} weights used (renormalized to sum=1): "
          f"{ {c: round(w, 3) for c, w in renorm_weights.items()} }")

    index = sum(normalized[c] * renorm_weights[c] for c in normalized)
    return index, pd.DataFrame(normalized), renorm_weights


def resolve_demand_floor(explicit_value, infrastructure, placeholder, label):
    """
    Resolve a Tier 2 demand floor, printing its provenance:
      1. explicit CONFIG value (analyst/team decision) if not None;
      2. calibration CSV row (revealed-deployment percentile p{FLOOR_PERCENTILE});
      3. literature-anchored placeholder, with a loud warning to run Stage 2b.
    """
    if explicit_value is not None:
        print(f"  {label}: {explicit_value:,.0f}  [source: explicit CONFIG value]")
        return float(explicit_value), "explicit_config"

    if os.path.exists(CALIBRATION_CSV):
        calib = pd.read_csv(CALIBRATION_CSV)
        row = calib[calib["infrastructure"] == infrastructure]
        pcol = f"p{FLOOR_PERCENTILE:02d}"
        if len(row) == 1 and pcol in row.columns and pd.notna(row.iloc[0][pcol]):
            value = float(row.iloc[0][pcol])
            n_sites = int(row.iloc[0].get("n_sites", 0))
            print(f"  {label}: {value:,.0f}  [source: calibration CSV, {pcol} of "
                  f"{n_sites} existing {infrastructure} sites - {CALIBRATION_CSV}]")
            return value, f"calibration_{pcol}"
        print(f"  *** calibration CSV found but no usable '{infrastructure}' row/"
              f"{pcol} column - falling back to placeholder. ***")

    print(f"  *** WARNING: {label} = {placeholder:,.0f} [source: literature-anchored "
          f"PLACEHOLDER]. Run stage2b_demand_floor_calibration.py to replace it with "
          f"an in-country calibrated value before decisions rest on E4/C3 counts. ***")
    return float(placeholder), "literature_placeholder"


def get_map_center(gdf):
    if len(gdf) == 0:
        return [6.4281, -9.4295]  # Liberia, approximate center
    centroids = gdf.to_crs(CRS_METRIC).centroid.to_crs(CRS_WGS84)
    return [centroids.y.mean(), centroids.x.mean()]


def save_map(m, filename):
    out_path = os.path.join(OUTPUT_DIR, filename)
    m.save(out_path)
    print(f"Saved map: {out_path}")
    display(m)
    return out_path


def add_legend(m, title, items):
    rows = "".join(f'<span style="color:{color};">&#9679;</span> {label}<br>' for label, color in items)
    html = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; width: 300px;
    background-color: white; z-index:9999; font-size:13px; border:2px solid grey; padding: 10px;">
    <b>{title}</b><br><br>
    {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


# %%
# =========================================================
# 3. LOAD STAGE 2A INDICATORS (includes Stage 1 columns already merged in)
# =========================================================

indicators = gpd.read_file(INDICATORS_GPKG, layer=INDICATORS_LAYER)
print(f"Loaded {len(indicators):,} institutions with indicators from {INDICATORS_GPKG}")
print(indicators.columns.tolist())


# %%
# =========================================================
# 4. MARKET VIABILITY INDEX (MVI) AND PHYSICAL CONSTRAINT INDEX (PCI)
# =========================================================

# Road-isolation composite: 0 (road within 2km) .. PCI_ROAD_BASE_WEIGHT
# (no road within 2km but one within 5km) .. PCI_ROAD_BASE_WEIGHT +
# PCI_ROAD_SEVERE_BONUS (no road even within 5km), scaled to [0,1] so it can
# be fed through the same generic weighting helper as the other components.
road_isolation_weight_budget = PCI_ROAD_BASE_WEIGHT + PCI_ROAD_SEVERE_BONUS
indicators["road_isolation_norm"] = (
    PCI_ROAD_BASE_WEIGHT * indicators["no_road_2km"].astype(float)
    + PCI_ROAD_SEVERE_BONUS * indicators["no_road_5km"].astype(float)
) / road_isolation_weight_budget

pci_weights = dict(PCI_WEIGHTS)
if USE_SECURITY and "security_risk_fallback" in indicators.columns:
    indicators["security_risk_ordinal"] = indicators["security_risk_fallback"].map(SECURITY_RISK_ORDINAL_MAP)
    total_before = sum(s["weight"] for s in pci_weights.values())
    new_total = total_before + SECURITY_WEIGHT
    pci_weights = {c: {**s, "weight": s["weight"] / new_total} for c, s in pci_weights.items()}
    pci_weights["security_risk_ordinal"] = {"weight": SECURITY_WEIGHT / new_total, "log": False, "cap": None}
    print(f"USE_SECURITY=True: added security_risk_ordinal (renormalized PCI weights, "
          f"sum={sum(s['weight'] for s in pci_weights.values()):.3f}).")

print("\nBuilding MVI...")
indicators["MVI"], mvi_components, mvi_weights_used = build_weighted_index(indicators, MVI_WEIGHTS, "MVI")
print("Building PCI...")
indicators["PCI"], pci_components, pci_weights_used = build_weighted_index(indicators, pci_weights, "PCI")

print(f"\nMVI: min={indicators['MVI'].min():.3f} median={indicators['MVI'].median():.3f} max={indicators['MVI'].max():.3f}")
print(f"PCI: min={indicators['PCI'].min():.3f} median={indicators['PCI'].median():.3f} max={indicators['PCI'].max():.3f}")


# %%
# =========================================================
# 5. TIER 2 DEMAND FLOORS (replaces the Rev. 2 M1-M4 median-split gate)
# =========================================================
# The M1-M4 segmentation is removed entirely in Rev. 3: gating new-build
# rungs on "MVI >= the global median" filtered out 100% of gap sites in the
# Liberia run (gap sites are remote/low-density by construction; the median
# is computed over all institutions including urban ones), and a median is a
# statistical artifact with no economic meaning - exactly half the country
# is always "high viability" regardless of whether anything is viable.
# The floors below are absolute exclusion screens on the RAW indicators
# (not on the sample-relative MVI), calibrated from Liberia's own existing
# deployments and cross-checked against ESMAP 2022 / GSMA Liberia evidence.

print("\nResolving Tier 2 demand floors...")
MIN_POP_MINIGRID_2KM, minigrid_floor_source = resolve_demand_floor(
    MIN_POP_MINIGRID_2KM, "minigrid", PLACEHOLDER_MIN_POP_MINIGRID_2KM,
    "MIN_POP_MINIGRID_2KM (E4 new mini-grid floor, pop_2km)",
)
MIN_POP_TOWER_10KM, tower_floor_source = resolve_demand_floor(
    MIN_POP_TOWER_10KM, "tower", PLACEHOLDER_MIN_POP_TOWER_10KM,
    "MIN_POP_TOWER_10KM (C3 new tower floor, pop_10km)",
)

indicators["meets_minigrid_demand_floor"] = (
    indicators["pop_2km"].fillna(0) >= MIN_POP_MINIGRID_2KM
)
indicators["meets_tower_demand_floor"] = (
    indicators["pop_10km"].fillna(0) >= MIN_POP_TOWER_10KM
)
print(f"  meets_minigrid_demand_floor: {indicators['meets_minigrid_demand_floor'].sum():,} "
      f"of {len(indicators):,} institutions")
print(f"  meets_tower_demand_floor:    {indicators['meets_tower_demand_floor'].sum():,} "
      f"of {len(indicators):,} institutions")


# %%
# =========================================================
# 6. CONNECTIVITY LADDER (C0-C5) AND ENERGY LADDER (E0-E5)
# =========================================================
# First match wins, cheapest credible option first. Tier 1 rungs (E0-E3,
# C0-C2) are pure distance/geometry rules; the new-build rungs (E4, C3) are
# additionally gated by the Tier 2 demand floors from Section 5. Already-
# served sites collapse to rung 0 - they don't need an extension-feasibility
# read, just QoS/verification.

dist_grid_km = indicators[
    ["dist_to_distribution_line_km", "dist_to_distribution_transformer_km"]
].min(axis=1)
dist_tower_km = indicators[
    ["dist_to_nearest_2g_tower_km", "dist_to_nearest_3g_tower_km", "dist_to_nearest_4g_tower_km"]
].min(axis=1)


def build_energy_ladder(df, floor_minigrid):
    """E0-E5, first match wins. floor_minigrid parameterized for sensitivity runs."""
    energy_gap = ~df["energy_access_proxy"].astype(bool)
    e0 = ~energy_gap
    e1 = energy_gap & df["near_nea_grid_expansion_transformer_5km"].astype(bool)
    e2 = (
        energy_gap & ~e1
        & dist_grid_km.gt(ENERGY_ACCESS_LINE_KM) & dist_grid_km.le(GRID_EXTENSION_MAX_KM)
    )
    e3 = (
        energy_gap & ~e1 & ~e2
        & df["dist_to_minigrid_km"].gt(ENERGY_ACCESS_MINIGRID_KM)
        & df["dist_to_minigrid_km"].le(MINIGRID_EXTENSION_MAX_KM)
    )
    # Tier 2 gate: new mini-grid only where near-field demand clears the floor
    e4 = (
        energy_gap & ~e1 & ~e2 & ~e3
        & (df["pop_2km"].fillna(0) >= floor_minigrid)
    )
    ladder = np.select([e0, e1, e2, e3, e4], ["E0", "E1", "E2", "E3", "E4"], default="E5")
    ladder = pd.Series(ladder, index=df.index)
    ladder.loc[~energy_gap] = "E0"
    return ladder


def build_connectivity_ladder(df, floor_tower):
    """
    C0-C5, first match wins, CHEAPEST FIRST: C2 (extension from an existing
    tower) is evaluated BEFORE C3 (new tower). Rev. 2 evaluated C3 first,
    which contradicted the cheapest-credible-option principle and the
    methodology document; fixed here.
    """
    digital_gap = ~df["digital_access_proxy"].astype(bool)
    # Buildability screen (5 km): one band beyond the RAI/SDG 9.1.1 personal-
    # access standard (2 km to an all-season road, ~20-25 min walk), matching
    # OnSSET's most-favorable distance-to-roads class (0-5 km) and Stage 2a's
    # severe-isolation flag. See module docstring for full provenance.
    has_road_5km = ~df["no_road_5km"].astype(bool)
    c0 = df["has_4g_coverage_proxy"].astype(bool)
    c1 = ~c0 & (
        df["has_2g_coverage_proxy"].astype(bool) | df["has_3g_coverage_proxy"].astype(bool)
    )
    c2 = digital_gap & ~c0 & ~c1 & dist_tower_km.le(TOWER_EXTENSION_MAX_KM)
    # Tier 2 gate: new tower only where catchment demand clears the floor
    # AND the site is physically reachable (road within 5 km, a Tier 1 rule).
    c3 = (
        digital_gap & ~c0 & ~c1 & ~c2
        & (df["pop_10km"].fillna(0) >= floor_tower)
        & has_road_5km
    )
    # Floor not met but accessible -> community WiFi (satellite/microwave backhaul)
    c4 = digital_gap & ~c0 & ~c1 & ~c2 & ~c3 & has_road_5km
    # Remaining: no road within 5 km (frontier) -> institution-level satellite
    ladder = np.select([c0, c1, c2, c3, c4], ["C0", "C1", "C2", "C3", "C4"], default="C5")
    ladder = pd.Series(ladder, index=df.index)
    # Safety net: any covered site that somehow missed c0/c1 collapses to C0.
    # NOTE - Rev. 2 forced ALL covered sites to C0 here, which silently
    # clobbered C1 (2G/3G -> RAN upgrade) back to C0 so C1 could never fire;
    # fixed to only correct covered sites sitting on an unserved rung.
    covered = ~digital_gap
    ladder.loc[covered & ~ladder.isin(["C0", "C1"])] = "C0"
    return ladder


indicators["energy_ladder"] = build_energy_ladder(indicators, MIN_POP_MINIGRID_2KM)
indicators["connectivity_ladder"] = build_connectivity_ladder(indicators, MIN_POP_TOWER_10KM)

# Sanity assertions: served sites never land on unserved rungs and vice versa.
assert (indicators.loc[indicators["energy_access_proxy"].astype(bool), "energy_ladder"] == "E0").all()
assert (indicators.loc[indicators["energy_ladder"] != "E0", "energy_access_proxy"].astype(bool) == False).all()
assert (indicators.loc[indicators["digital_access_proxy"].astype(bool), "connectivity_ladder"].isin(["C0", "C1"])).all()

ENERGY_LADDER_LABELS = {
    "E0": "Already energy-served (access proxy true) - verification only",
    "E1": "Planned NEA grid-expansion transformer within 5km - coordinate with rollout, avoid duplicate investment",
    "E2": f"Distribution line/transformer {ENERGY_ACCESS_LINE_KM}-{GRID_EXTENSION_MAX_KM}km away - grid extension candidate",
    "E3": f"Mini-grid {ENERGY_ACCESS_MINIGRID_KM}-{MINIGRID_EXTENSION_MAX_KM}km away - mini-grid extension candidate",
    "E4": f"No extendable infrastructure nearby, pop_2km >= demand floor ({MIN_POP_MINIGRID_2KM:,.0f}) - new standalone mini-grid",
    "E5": "No extendable infrastructure nearby, below demand floor - standalone solar EaaS",
}
CONNECTIVITY_LADDER_LABELS = {
    "C0": "Already 4G-covered - QoS monitoring only",
    "C1": "2G/3G only, no 4G - existing-site RAN upgrade (low cost)",
    "C2": f"Nearest tower within {TOWER_EXTENSION_MAX_KM}km - coverage extension/densification",
    "C3": f"Unserved, pop_10km >= demand floor ({MIN_POP_TOWER_10KM:,.0f}), road access - new tower (ABC+ anchor candidate)",
    "C4": "Unserved, below demand floor, road access - community WiFi via satellite/microwave backhaul",
    "C5": "Unserved, no road within 5km (frontier) - standalone institutional VSAT/LEO",
}

print(indicators["connectivity_ladder"].value_counts().sort_index())
print(indicators["energy_ladder"].value_counts().sort_index())

# --- Floor sensitivity report: how fragile are E4/C3 counts to the floor? ---
print("\n--- Demand-floor sensitivity (E4 / C3 counts under scaled floors) ---")
sens_rows = []
for factor in FLOOR_SENSITIVITY_FACTORS:
    e_l = build_energy_ladder(indicators, MIN_POP_MINIGRID_2KM * factor)
    c_l = build_connectivity_ladder(indicators, MIN_POP_TOWER_10KM * factor)
    sens_rows.append({
        "floor_factor": f"x{factor}",
        "minigrid_floor": round(MIN_POP_MINIGRID_2KM * factor),
        "n_E4": int((e_l == "E4").sum()),
        "n_E5": int((e_l == "E5").sum()),
        "tower_floor": round(MIN_POP_TOWER_10KM * factor),
        "n_C3": int((c_l == "C3").sum()),
        "n_C4": int((c_l == "C4").sum()),
    })
sensitivity_df = pd.DataFrame(sens_rows)
print(sensitivity_df.to_string(index=False))
sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, "demand_floor_sensitivity.csv"), index=False)
print("If E4/C3 counts swing widely across x0.5-x1.5, prefer a more conservative "
      "percentile (FLOOR_PERCENTILE=25) or agree floors with the sector teams.")

# --- DRE Atlas cross-check (NEVER a score/gate input) ---
# If the floors are doing their job, E4 sites should sit near visibly larger
# settlements than E5 sites (and C3 > C4/C5) on this INDEPENDENT dataset.
if "dreatlas_pop_nearest" in indicators.columns:
    print("\n--- Cross-check: median DRE Atlas nearest-settlement population by rung ---")
    print("(independent settlement data; verification only, never enters any score)")
    print(indicators.groupby("energy_ladder")["dreatlas_pop_nearest"].median().round(0))
    print(indicators.groupby("connectivity_ladder")["dreatlas_pop_nearest"].median().round(0))


# %%
# =========================================================
# 7. JOINT DELIVERY TYPE (Type A/C) AND DELIVERY MODEL
# =========================================================
# Keyed directly off Stage 1's combined_access_status gap (what's actually
# missing) rather than an absolute ladder-rung cutoff. Mapping mirrors the
# concept note's matching matrix (Table 4) one-to-one.
# NOTE on Stage 1 label semantics, to avoid a classic misread:
#   "Digital only"  = HAS digital, MISSING energy  -> Type A EaaS leg
#   "Energy only"   = HAS energy,  MISSING digital -> Type C co-location (C3)
#                                                     or Type A CaaS leg

dual_gap = indicators["combined_access_status"] == "No energy or digital access"
missing_energy_only = indicators["combined_access_status"] == "Digital only"   # has digital, missing energy
missing_digital_only = indicators["combined_access_status"] == "Energy only"   # has energy, missing digital
no_gap = indicators["combined_access_status"] == "Energy + Digital access"

is_c3 = indicators["connectivity_ladder"] == "C3"
is_e3_or_e4 = indicators["energy_ladder"].isin(["E3", "E4"])

cond_abc_plus = dual_gap & is_c3 & is_e3_or_e4
cond_ecaas = dual_gap & ~cond_abc_plus
cond_eaas = missing_energy_only
cond_colocation = missing_digital_only & is_c3
cond_caas = missing_digital_only & ~is_c3

indicators["joint_delivery_type"] = np.select(
    [cond_abc_plus, cond_ecaas, cond_eaas, cond_colocation, cond_caas, no_gap],
    ["Type C", "Type A", "Type A", "Type C", "Type A", "Served"],
    default="Served",
)
indicators["delivery_model"] = np.select(
    [cond_abc_plus, cond_ecaas, cond_eaas, cond_colocation, cond_caas, no_gap],
    [
        "ABC+ (anchor-bundle-community)",
        "ECaaS (energy+connectivity-as-a-service, dual gap)",
        "EaaS leg (energy-as-a-service; digital already present)",
        "Co-location (new tower at existing energy site)",
        "CaaS leg (connectivity-as-a-service; energy already present)",
        "Already served - no delivery model needed",
    ],
    default="Already served - no delivery model needed",
)

JOINT_TYPE_LABELS = {
    "Type A": "As-a-service leg: ECaaS (dual gap, no bundling anchor) / EaaS (energy gap only) / CaaS (digital gap only)",
    "Type C": "Bundled: ABC+ anchor-bundle-community (dual gap w/ viable anchor+tower) or co-location (new tower at existing energy site)",
    "Served": "Energy + Digital access already present - no delivery model needed",
}
print(indicators["joint_delivery_type"].value_counts())
print(indicators["delivery_model"].value_counts())

# --- Type B demand-side overlay flag (concept note Table 4 caption) ---
# NOT an alternative to the site-level options above: marks institutions
# where a household/MSME solar-kit + digital-device affordability window
# (toolkit Type B) should run IN PARALLEL with the site contract. Rule:
# residual access gap at the site, OR low-wealth catchment. If RWI is
# unavailable (manual input not yet placed), the wealth half is skipped
# with a warning and the flag reduces to "any residual gap".
has_residual_gap = ~no_gap
if indicators["rwi_popweighted_10km"].notna().any():
    low_wealth_catchment = indicators["rwi_popweighted_10km"] <= TYPE_B_RWI_THRESHOLD
else:
    print("  *** WARNING: rwi_popweighted_10km all-NaN - Type B flag uses the "
          "residual-gap condition only until the RWI file is placed. ***")
    low_wealth_catchment = pd.Series(False, index=indicators.index)
indicators["type_b_window_flag"] = has_residual_gap | low_wealth_catchment
print(f"type_b_window_flag: {indicators['type_b_window_flag'].sum():,} of "
      f"{len(indicators):,} institutions "
      f"(residual gap: {has_residual_gap.sum():,}; "
      f"low-wealth catchment RWI<={TYPE_B_RWI_THRESHOLD}: {low_wealth_catchment.sum():,})")

# Rev. 3 note: the Rev. 2 "known limitation" (M1/M2 median gate filtering out
# 100% of gap sites, collapsing Type C to zero) is RESOLVED by the Tier 2
# demand floors - the gate is now an absolute demand test on the raw
# indicators, independent of where the country-wide median happens to sit.
# If Type C is still zero after calibration, that is now a substantive
# finding about demand around gap sites, not a construction artifact; check
# the sensitivity report and the DRE cross-check before treating it as final.
if (indicators["joint_delivery_type"] == "Type C").sum() == 0:
    print(
        "\n*** NOTE: 0 Type C (ABC+ / co-location) sites under the current demand "
        "floors. Inspect demand_floor_sensitivity.csv and the DRE cross-check "
        "before concluding no site supports a new tower. ***"
    )


# %%
# =========================================================
# 8. PRIORITY SCORE
# =========================================================

beneficiary_scale = _percentile_normalize(np.log1p(indicators["pop_2km"].clip(lower=0)))
beneficiary_scale = beneficiary_scale.fillna(beneficiary_scale.median())
access_gap = indicators["combined_access_status"].map(ACCESS_GAP_SCORE).fillna(0.5)
constraint_ease = 1 - indicators["PCI"]

indicators["priority_score"] = (
    PRIORITY_WEIGHTS["beneficiary_scale"] * beneficiary_scale
    + PRIORITY_WEIGHTS["access_gap"] * access_gap
    + PRIORITY_WEIGHTS["market_viability"] * indicators["MVI"]
    + PRIORITY_WEIGHTS["constraint_ease"] * constraint_ease
)
indicators["priority_rank"] = indicators["priority_score"].rank(ascending=False, method="min").astype(int)
indicators = indicators.sort_values("priority_rank").reset_index(drop=True)

print(indicators[["institution_name", "priority_rank", "priority_score",
                  "energy_ladder", "connectivity_ladder", "joint_delivery_type"]].head(10))

# Rung-internal ordering = filter the global ranking by rung (no separate score):
# e.g. which E4 sites to appraise first for a new mini-grid.
print("\nTop 5 E4 (new mini-grid) sites by priority rank:")
e4_sites = indicators[indicators["energy_ladder"] == "E4"]
print(e4_sites[["institution_name", "priority_rank", "pop_2km"]].head(5))
print("\nTop 5 C3 (new tower) sites by priority rank:")
c3_sites = indicators[indicators["connectivity_ladder"] == "C3"]
print(c3_sites[["institution_name", "priority_rank", "pop_10km"]].head(5))


# %%
# =========================================================
# 9. PROCUREMENT LOTS (geographic buffer clustering)
# =========================================================

indicators_m = indicators.to_crs(CRS_METRIC)
half_radius_m = (LOT_CLUSTER_RADIUS_KM / 2) * 1000
buffered = indicators_m.geometry.buffer(half_radius_m)
merged_geom = unary_union(buffered.values)
cluster_polys = [merged_geom] if merged_geom.geom_type == "Polygon" else list(merged_geom.geoms)

clusters_gdf = gpd.GeoDataFrame(
    {"cluster_id": range(len(cluster_polys))}, geometry=cluster_polys, crs=CRS_METRIC
)
joined_clusters = gpd.sjoin(
    indicators_m[["institution_id", "geometry"]], clusters_gdf, predicate="within", how="left"
).set_index("institution_id")["cluster_id"]
indicators["lot_cluster_id"] = indicators["institution_id"].map(joined_clusters)


def assign_procurement_lot_ids(df, cluster_col, max_size):
    """Split any cluster bigger than max_size into multiple same-cluster lots."""
    lot_ids = pd.Series(index=df.index, dtype=object)
    for cluster_id, group in df.groupby(cluster_col):
        n_sub = int(np.ceil(len(group) / max_size))
        for i, idx_chunk in enumerate(np.array_split(group.index, n_sub)):
            lot_ids.loc[idx_chunk] = f"LOT-{int(cluster_id):04d}-{i + 1}"
    return lot_ids


indicators["procurement_lot_id"] = assign_procurement_lot_ids(indicators, "lot_cluster_id", MAX_INSTITUTIONS_PER_LOT)
print(f"Procurement lots: {indicators['procurement_lot_id'].nunique():,} lots "
      f"across {indicators['lot_cluster_id'].nunique():,} geographic clusters "
      f"(max {MAX_INSTITUTIONS_PER_LOT} institutions/lot).")


# %%
# =========================================================
# 10. MAPS
# =========================================================

# --- Ladder map (replaces the Rev. 2 M1-M4 segment map): two toggleable
# layers, one per ladder, with the demand-floor evidence in every popup. ---
m_ladder = folium.Map(location=get_map_center(indicators), zoom_start=7, tiles="CartoDB positron")
energy_colors = {"E0": "green", "E1": "teal", "E2": "blue", "E3": "cadetblue",
                 "E4": "orange", "E5": "red"}
conn_colors = {"C0": "green", "C1": "teal", "C2": "blue",
               "C3": "orange", "C4": "purple", "C5": "red"}
energy_group = folium.FeatureGroup(name="Energy ladder (E0-E5)", show=True).add_to(m_ladder)
conn_group = folium.FeatureGroup(name="Connectivity ladder (C0-C5)", show=False).add_to(m_ladder)
for _, row in indicators.iterrows():
    popup = f"""
    <b>{row['institution_name']}</b><br>
    Energy ladder: {row['energy_ladder']} - {ENERGY_LADDER_LABELS.get(row['energy_ladder'], '')}<br>
    Connectivity ladder: {row['connectivity_ladder']} - {CONNECTIVITY_LADDER_LABELS.get(row['connectivity_ladder'], '')}<br>
    pop_2km: {row['pop_2km']:.0f} (mini-grid floor {MIN_POP_MINIGRID_2KM:,.0f}:
    {'met' if row['meets_minigrid_demand_floor'] else 'not met'})<br>
    pop_10km: {row['pop_10km']:.0f} (tower floor {MIN_POP_TOWER_10KM:,.0f}:
    {'met' if row['meets_tower_demand_floor'] else 'not met'})<br>
    Priority rank: {row['priority_rank']} (score {row['priority_score']:.2f})
    """
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=energy_colors.get(row["energy_ladder"], "gray"), fill=True,
        fill_color=energy_colors.get(row["energy_ladder"], "gray"), fill_opacity=0.75,
        popup=folium.Popup(popup, max_width=380),
    ).add_to(energy_group)
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=conn_colors.get(row["connectivity_ladder"], "gray"), fill=True,
        fill_color=conn_colors.get(row["connectivity_ladder"], "gray"), fill_opacity=0.75,
        popup=folium.Popup(popup, max_width=380),
    ).add_to(conn_group)
add_legend(m_ladder, "Ladder rungs (toggle layers)",
           [(f"Energy {k}", v) for k, v in energy_colors.items()]
           + [(f"Connectivity {k}", v) for k, v in conn_colors.items()])
folium.LayerControl(collapsed=False).add_to(m_ladder)
save_map(m_ladder, "05_institution_extension_ladders.html")

# --- Joint delivery type map ---
m_type = folium.Map(location=get_map_center(indicators), zoom_start=7, tiles="CartoDB positron")
type_colors = {"Type A": "purple", "Type C": "orange", "Served": "green"}
type_group = folium.FeatureGroup(name="Institutions by joint delivery type").add_to(m_type)
for _, row in indicators.iterrows():
    popup = f"""
    <b>{row['institution_name']}</b><br>
    Joint delivery type: {row['joint_delivery_type']} - {row['delivery_model']}<br>
    Type B window (household/MSME affordability overlay): {'yes' if row['type_b_window_flag'] else 'no'}<br>
    Connectivity ladder: {row['connectivity_ladder']} | Energy ladder: {row['energy_ladder']}<br>
    Access gap: {row['combined_access_status']}<br>
    Priority rank: {row['priority_rank']} (score {row['priority_score']:.2f})<br>
    Procurement lot: {row['procurement_lot_id']}
    """
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=type_colors.get(row["joint_delivery_type"], "gray"), fill=True,
        fill_color=type_colors.get(row["joint_delivery_type"], "gray"), fill_opacity=0.75,
        popup=folium.Popup(popup, max_width=350),
    ).add_to(type_group)
add_legend(m_type, "Joint Delivery Type", list(type_colors.items()))
folium.LayerControl(collapsed=False).add_to(m_type)
save_map(m_type, "06_institution_joint_delivery_type.html")


# %%
# =========================================================
# 11. SAVE OUTPUTS
# =========================================================

out_gpkg = os.path.join(OUTPUT_DIR, "institution_delivery_matching.gpkg")
indicators.to_file(out_gpkg, layer="delivery_matching", driver="GPKG")

out_df = indicators.copy()
out_df["longitude"] = out_df.geometry.x
out_df["latitude"] = out_df.geometry.y
out_df = pd.DataFrame(out_df.drop(columns="geometry"))
out_csv = os.path.join(OUTPUT_DIR, "institution_delivery_matching.csv")
out_df.to_csv(out_csv, index=False)

print("Saved:")
print(out_gpkg)
print(out_csv)

print("\nAccess typology x Joint Delivery Type crosstab (check that GAP sites, "
      "not already-served sites, drive Type A/C assignments):")
print(pd.crosstab(indicators["combined_access_status"], indicators["joint_delivery_type"]))

print("\nType B overlay by joint delivery type (affordability windows run in "
      "parallel with, not instead of, the site contract):")
print(pd.crosstab(indicators["joint_delivery_type"], indicators["type_b_window_flag"]))

print("\nDemand floors used in this run (record in the methodology annex):")
print(f"  MIN_POP_MINIGRID_2KM = {MIN_POP_MINIGRID_2KM:,.0f}  [{minigrid_floor_source}]")
print(f"  MIN_POP_TOWER_10KM   = {MIN_POP_TOWER_10KM:,.0f}  [{tower_floor_source}]")
