# -*- coding: utf-8 -*-
"""
Stage 2 — Delivery Matching (Liberia)

Reads institution_indicators.gpkg (Stage 2a's output; institution-level
buffer/distance indicators + every Stage 1 column carried forward) and scores
each institution with no settlement join anywhere in the scoring path.

Computes, per institution:
    MVI  (Market Viability Index, 0-1)   demand-side composite; see MVI_WEIGHTS.
    PCI  (Physical Constraint Index, 0-1) supply-side composite; see PCI_WEIGHTS.
    segment (M1-M4)                       2x2 median-split of MVI x PCI.
    connectivity_ladder (C0-C5)           for DIGITAL-gap sites: how close to
                                           extendable telecom infrastructure
                                           (existing tower / road / anchor
                                           viability), NOT how well-served the
                                           site already is - see Section 6.
    energy_ladder (E0-E5)                 for ENERGY-gap sites: how close to
                                           extendable energy infrastructure
                                           (planned rollout / grid / mini-grid
                                           / standalone) - see Section 6.
    joint_delivery_type (Type A/C),
    delivery_model                         keyed off Stage 1's combined_access_status
                                           gap (dual/energy-only/digital-only/none),
                                           not an absolute ladder-rung cutoff -
                                           see Section 7.
    priority_score / priority_rank         beneficiary scale (pop_2km) + access
                                           gap (Stage 1 combined_access_status)
                                           + MVI + (1 - PCI); see PRIORITY_WEIGHTS.
    lot_cluster_id / procurement_lot_id    geographic buffer-clustering for
                                           bundled procurement lots.

The MVI/PCI/ladder/segmentation/priority/lotting design in this script has no
prior settlement-based version to preserve (none existed on disk); it is a
new, from-scratch design built for the institution-level architecture, with
every threshold/weight/radius in the CONFIG block below so it can be retuned
without touching the logic.

KNOWN LIMITATION (accepted as-is, not a bug): C3 / E4 / "Type C" (ABC+ /
co-location) require a gap site to also be segment M1/M2, i.e. MVI >= the
GLOBAL median across all institutions. Because MVI is built from population/
building/RWI density and gap sites are disproportionately remote and
low-density by construction, this can filter out most or all gap sites (in
the current Liberia run: 100%, so Type C is 0 institutions). Left unchanged
per product decision; a gap-local MVI threshold would surface relative
anchors within the gap population instead if this needs revisiting later.

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
# 5. MARKET SEGMENTATION (M1-M4): median-split of MVI x PCI, over institutions
# =========================================================

mvi_median = indicators["MVI"].median()
pci_median = indicators["PCI"].median()
high_mvi = indicators["MVI"] >= mvi_median
high_pci = indicators["PCI"] >= pci_median

segment_conditions = [high_mvi & ~high_pci, high_mvi & high_pci, ~high_mvi & ~high_pci, ~high_mvi & high_pci]
segment_choices = ["M1", "M2", "M3", "M4"]
indicators["segment"] = np.select(segment_conditions, segment_choices, default="Unclassified")

SEGMENT_LABELS = {
    "M1": "High viability / Low constraint - best fit for commercial delivery",
    "M2": "High viability / High constraint - viable demand, hard to reach (premium/leapfrog)",
    "M4": "Low viability / High constraint - hardest segment (deprioritize or last-mile subsidy)",
    "M3": "Low viability / Low constraint - easy to reach, weak demand (bundle with anchor)",
}
print(indicators["segment"].value_counts())


# %%
# =========================================================
# 6. CONNECTIVITY LADDER (C0-C5) AND ENERGY LADDER (E0-E5)
# =========================================================
# Both ladders only produce a real gradient for GAP sites (energy_access_proxy
# / digital_access_proxy == False). Already-served sites collapse to rung 0 -
# they don't need an extension-feasibility read, just QoS/verification.
# high_mvi (MVI >= median) is already computed in Section 5.

dist_grid_km = indicators[
    ["dist_to_distribution_line_km", "dist_to_distribution_transformer_km"]
].min(axis=1)
dist_tower_km = indicators[
    ["dist_to_nearest_2g_tower_km", "dist_to_nearest_3g_tower_km", "dist_to_nearest_4g_tower_km"]
].min(axis=1)

# --- Energy ladder ---
energy_gap = ~indicators["energy_access_proxy"].astype(bool)
cond_e0 = ~energy_gap
cond_e1 = energy_gap & indicators["near_nea_grid_expansion_transformer_5km"].astype(bool)
cond_e2 = (
    energy_gap & ~cond_e1
    & dist_grid_km.gt(ENERGY_ACCESS_LINE_KM) & dist_grid_km.le(GRID_EXTENSION_MAX_KM)
)
cond_e3 = (
    energy_gap & ~cond_e1 & ~cond_e2
    & indicators["dist_to_minigrid_km"].gt(ENERGY_ACCESS_MINIGRID_KM)
    & indicators["dist_to_minigrid_km"].le(MINIGRID_EXTENSION_MAX_KM)
)
cond_e4 = energy_gap & ~cond_e1 & ~cond_e2 & ~cond_e3 & high_mvi
# remaining energy-gap sites (low MVI, nothing extendable nearby) -> E5

indicators["energy_ladder"] = np.select(
    [cond_e0, cond_e1, cond_e2, cond_e3, cond_e4],
    ["E0", "E1", "E2", "E3", "E4"],
    default="E5",
)
indicators.loc[~energy_gap, "energy_ladder"] = "E0"

ENERGY_LADDER_LABELS = {
    "E0": "Already energy-served (access proxy true) - verification only",
    "E1": "Planned NEA grid-expansion transformer within 5km - coordinate with rollout, avoid duplicate investment",
    "E2": f"Distribution line/transformer {ENERGY_ACCESS_LINE_KM}-{GRID_EXTENSION_MAX_KM}km away - grid extension candidate",
    "E3": f"Mini-grid {ENERGY_ACCESS_MINIGRID_KM}-{MINIGRID_EXTENSION_MAX_KM}km away - mini-grid extension candidate",
    "E4": "No extendable infrastructure nearby, high viability (M1/M2) - new standalone mini-grid",
    "E5": "No extendable infrastructure nearby, low viability (M3/M4) - standalone solar EaaS",
}

# --- Connectivity ladder ---
digital_gap = ~indicators["digital_access_proxy"].astype(bool)
cond_c0 = indicators["has_4g_coverage_proxy"].astype(bool)
cond_c1 = ~cond_c0 & (
    indicators["has_2g_coverage_proxy"].astype(bool) | indicators["has_3g_coverage_proxy"].astype(bool)
)
cond_c3 = digital_gap & ~cond_c0 & ~cond_c1 & high_mvi & ~indicators["no_road_5km"].astype(bool)
cond_c2 = digital_gap & ~cond_c0 & ~cond_c1 & ~cond_c3 & dist_tower_km.le(TOWER_EXTENSION_MAX_KM)
cond_c4 = digital_gap & ~cond_c0 & ~cond_c1 & ~cond_c3 & ~cond_c2 & (indicators["segment"] == "M3")
# remaining digital-gap sites (M4, no road within 5km) -> C5

indicators["connectivity_ladder"] = np.select(
    [cond_c0, cond_c1, cond_c2, cond_c3, cond_c4],
    ["C0", "C1", "C2", "C3", "C4"],
    default="C5",
)
indicators.loc[~digital_gap, "connectivity_ladder"] = "C0"

CONNECTIVITY_LADDER_LABELS = {
    "C0": "Already 4G-covered - QoS monitoring only",
    "C1": "2G/3G only, no 4G - existing-site RAN upgrade (low cost)",
    "C2": f"Nearest tower within {TOWER_EXTENSION_MAX_KM}km - coverage extension/densification",
    "C3": "Unserved, high viability (M1/M2), road access - new tower (ABC+ anchor candidate)",
    "C4": "Unserved, M3 segment - community WiFi via satellite backhaul",
    "C5": "Unserved, M4 segment, no road within 5km - standalone institutional VSAT/LEO",
}

print(indicators["connectivity_ladder"].value_counts().sort_index())
print(indicators["energy_ladder"].value_counts().sort_index())


# %%
# =========================================================
# 7. JOINT DELIVERY TYPE (Type A/C) AND DELIVERY MODEL
# =========================================================
# Keyed directly off Stage 1's combined_access_status gap (what's actually
# missing) rather than an absolute ladder-rung cutoff. The old cutoff-based
# Type B never fired: rwi_popweighted_10km drives MVI upward (positive MVI
# weight) and the old type_b_window_flag downward (poor-area threshold) in
# opposite directions on the same variable, so "MVI >= median AND RWI <= -0.3"
# was close to a self-contradiction once real RWI data was loaded (0 matches).

dual_gap = indicators["combined_access_status"] == "No energy or digital access"
energy_only_gap = indicators["combined_access_status"] == "Digital only"   # has digital, missing energy
digital_only_gap = indicators["combined_access_status"] == "Energy only"  # has energy, missing digital
no_gap = indicators["combined_access_status"] == "Energy + Digital access"

is_c3 = indicators["connectivity_ladder"] == "C3"
is_e3_or_e4 = indicators["energy_ladder"].isin(["E3", "E4"])

cond_abc_plus = dual_gap & is_c3 & is_e3_or_e4
cond_ecaas = dual_gap & ~cond_abc_plus
cond_eaas = energy_only_gap
cond_colocation = digital_only_gap & is_c3
cond_caas = digital_only_gap & ~is_c3

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

# KNOWN LIMITATION (left as-is per product decision, not a bug): C3/E4/ABC+
# require the gap site to also be M1/M2 (MVI >= the GLOBAL median across all
# 7,193 institutions). MVI is built from population/building/RWI density, and
# gap sites are disproportionately remote/low-density by construction, so the
# global-median bar can filter out ~all of them - in the current Liberia
# dataset it filters out 100% (0 gap sites qualify), collapsing ABC+/new-
# standalone-mini-grid to zero. A gap-local MVI threshold (e.g. median among
# gap sites only, or raw pop_2km rank) would surface relative anchors within
# the gap population instead - not implemented; flagging so it isn't mistaken
# for a bug if ABC+/E4 counts look suspiciously low or zero.
if (indicators["joint_delivery_type"] == "Type C").sum() == 0:
    print(
        "\n*** NOTE: 0 Type C (ABC+ / co-location) sites. This is the known MVI-vs-gap-site "
        "limitation above, not a bug - see the comment in Section 7. ***"
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

print(indicators[["institution_name", "priority_rank", "priority_score", "segment", "joint_delivery_type"]].head(10))


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

# --- Segment map (institutions colored by M1-M4, replaces old settlement-polygon map) ---
m_segment = folium.Map(location=get_map_center(indicators), zoom_start=7, tiles="CartoDB positron")
segment_colors = {"M1": "green", "M2": "orange", "M3": "blue", "M4": "red", "Unclassified": "gray"}
segment_group = folium.FeatureGroup(name="Institutions by market segment").add_to(m_segment)
for _, row in indicators.iterrows():
    popup = f"""
    <b>{row['institution_name']}</b><br>
    Segment: {row['segment']} - {SEGMENT_LABELS.get(row['segment'], '')}<br>
    MVI: {row['MVI']:.2f} | PCI: {row['PCI']:.2f}<br>
    pop_2km: {row['pop_2km']:.0f} | pop_10km: {row['pop_10km']:.0f}<br>
    RWI (10km, pop-weighted): {row['rwi_popweighted_10km']:.2f} ({row['rwi_method']})<br>
    Priority rank: {row['priority_rank']} (score {row['priority_score']:.2f})
    """
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=segment_colors.get(row["segment"], "gray"), fill=True,
        fill_color=segment_colors.get(row["segment"], "gray"), fill_opacity=0.75,
        popup=folium.Popup(popup, max_width=350),
    ).add_to(segment_group)
add_legend(m_segment, "Market Segment", list(segment_colors.items()))
folium.LayerControl(collapsed=False).add_to(m_segment)
save_map(m_segment, "05_institution_market_segment.html")

# --- Joint delivery type map ---
m_type = folium.Map(location=get_map_center(indicators), zoom_start=7, tiles="CartoDB positron")
type_colors = {"Type A": "purple", "Type C": "orange", "Served": "green"}
type_group = folium.FeatureGroup(name="Institutions by joint delivery type").add_to(m_type)
for _, row in indicators.iterrows():
    popup = f"""
    <b>{row['institution_name']}</b><br>
    Joint delivery type: {row['joint_delivery_type']} - {row['delivery_model']}<br>
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

print("\nSegment x Joint Delivery Type crosstab:")
print(pd.crosstab(indicators["segment"], indicators["joint_delivery_type"]))
