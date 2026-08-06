import os
import pandas as pd
import geopandas as gpd

# ══════════════════════════════════════════════════════════════
# 1) 경로 · 설정
# ══════════════════════════════════════════════════════════════
BASE     = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access"
GEP_CSV  = os.path.join(BASE, r"tz-3-scenarios-results\tz-3-0_0_0_0_0_0.csv")
CLUSTERS = os.path.join(BASE, r"tz-1\final_clusters.shp")
SCHOOLS  = os.path.join(BASE, r"merged\schools_TZA.csv")
OUT_DIR  = os.path.join(BASE, "energy_digital")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC_CRS = "EPSG:32736"      # 탄자니아 UTM 36S
NEAR_CLUSTER_MAX_M = 2000      # 이 거리 이내면 최근접 클러스터 값 사용, 넘으면 미전기화 확정

# ══════════════════════════════════════════════════════════════
# 2) GEP 시나리오 CSV — 기준연도(현황) 컬럼만
#    ※ 96개 시나리오 파일 모두 ElecStart 등 2020 컬럼은 동일하므로 하나만 읽으면 됨
# ══════════════════════════════════════════════════════════════
gep = pd.read_csv(GEP_CSV, low_memory=False)
print("GEP shape:", gep.shape)

keep = ["id", "ElecStart", "FinalElecCode2020", "ElecPopCalib",
        "Pop2020", "PopStartYear", "NightLights", "IsUrban",
        "CurrentMVLineDist", "TransformerDist", "GridDistCalibElec",
        "Prim", "Sec", "Unc", "Admin1"]
keep = [c for c in keep if c in gep.columns]
gep = gep[keep].copy()

pop_col = "Pop2020" if "Pop2020" in gep.columns else "PopStartYear"
if "ElecPopCalib" in gep.columns and pop_col in gep.columns:
    gep["elec_rate"] = (gep["ElecPopCalib"] / gep[pop_col].replace(0, pd.NA)).clip(0, 1)

TECH = {1: "Existing grid", 2: "Grid extension", 3: "Solar MG",
        5: "Hydro MG", 6: "Wind MG", 7: "Standalone PV", 99: "Not electrified"}
if "FinalElecCode2020" in gep.columns:
    gep["elec_tech_2020"] = gep["FinalElecCode2020"].map(TECH)

print("\nElecStart 분포:")
print(gep["ElecStart"].value_counts(dropna=False))

# ══════════════════════════════════════════════════════════════
# 3) 클러스터 shp + GEP 결합 (id 기준)
# ══════════════════════════════════════════════════════════════
clusters = gpd.read_file(CLUSTERS)
print("\nclusters:", clusters.shape, "| CRS:", clusters.crs)

id_col = "id" if "id" in clusters.columns else None
if id_col is None:
    raise SystemExit(f"id 컬럼 없음: {list(clusters.columns)}")

clusters = clusters[[id_col, "geometry"]].merge(gep, left_on=id_col, right_on="id", how="left")
print("GEP 붙은 비율:", f"{clusters['ElecStart'].notna().mean():.1%}")

if clusters.crs is None:
    clusters = clusters.set_crs("EPSG:4326")
clusters = clusters.to_crs("EPSG:4326")

ENERGY_COLS = [c for c in clusters.columns if c != "geometry"]

# ══════════════════════════════════════════════════════════════
# 4) 학교 포인트
# ══════════════════════════════════════════════════════════════
sch = pd.read_csv(SCHOOLS, low_memory=False)
sch = sch.dropna(subset=["latitude", "longitude"])
sch_gdf = gpd.GeoDataFrame(
    sch, geometry=gpd.points_from_xy(sch["longitude"], sch["latitude"]),
    crs="EPSG:4326")
print("\n학교 수:", len(sch_gdf))

# ══════════════════════════════════════════════════════════════
# 5) 공간조인 (within) → 경계중복 제거 → 최근접 보완
# ══════════════════════════════════════════════════════════════
joined = gpd.sjoin(sch_gdf, clusters, how="left", predicate="within")
joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])

before = len(joined)
joined = joined.drop_duplicates(subset="giga_id_school", keep="first")
print(f"경계 중복 제거: {before} → {len(joined)}행")
print(f"within 매칭률: {joined['ElecStart'].notna().mean():.1%}")

joined["match_type"]     = "within"
joined["cluster_dist_m"] = 0.0

miss_ids = joined.loc[joined["ElecStart"].isna(), "giga_id_school"]
print(f"폴리곤 밖 학교: {len(miss_ids)}개")

if len(miss_ids) > 0:
    miss_pts = sch_gdf[sch_gdf["giga_id_school"].isin(miss_ids)].to_crs(METRIC_CRS)
    near = gpd.sjoin_nearest(
        miss_pts, clusters.to_crs(METRIC_CRS),
        how="left", distance_col="cluster_dist_m")
    near = near.drop(columns=[c for c in ["index_right"] if c in near.columns])
    near = near.drop_duplicates(subset="giga_id_school", keep="first")

    print(f"  거리 중앙값: {near['cluster_dist_m'].median():.0f} m")
    print(f"  90퍼센타일: {near['cluster_dist_m'].quantile(0.9):.0f} m")
    print(f"  {NEAR_CLUSTER_MAX_M}m 이내: {(near['cluster_dist_m'] <= NEAR_CLUSTER_MAX_M).sum()}개")
    print(f"  {NEAR_CLUSTER_MAX_M}m 초과: {(near['cluster_dist_m'] >  NEAR_CLUSTER_MAX_M).sum()}개")

    joined = joined.set_index("giga_id_school")
    near   = near.set_index("giga_id_school")

    # (B) 가까운 학교 → 최근접 클러스터 값 사용
    close = near[near["cluster_dist_m"] <= NEAR_CLUSTER_MAX_M]
    for c in ENERGY_COLS:
        if c in close.columns:
            joined.loc[close.index, c] = close[c]
    joined.loc[close.index, "cluster_dist_m"] = close["cluster_dist_m"]
    joined.loc[close.index, "match_type"]     = "nearest"

    # (A) 먼 학교 → 미전기화로 확정
    far = near[near["cluster_dist_m"] > NEAR_CLUSTER_MAX_M]
    joined.loc[far.index, "ElecStart"]      = 0
    joined.loc[far.index, "elec_rate"]      = 0.0
    joined.loc[far.index, "elec_tech_2020"] = "Not electrified (isolated)"
    joined.loc[far.index, "cluster_dist_m"] = far["cluster_dist_m"]
    joined.loc[far.index, "match_type"]     = "isolated_no_elec"

    joined = joined.reset_index()

print(f"최종 ElecStart 채워진 비율: {joined['ElecStart'].notna().mean():.1%}")

# 신뢰도 라벨
joined["energy_confidence"] = pd.cut(
    joined["cluster_dist_m"],
    bins=[-1, 0, 500, NEAR_CLUSTER_MAX_M, float("inf")],
    labels=["within (high)", "≤500m (good)",
            f"≤{NEAR_CLUSTER_MAX_M}m (moderate)", f">{NEAR_CLUSTER_MAX_M}m (isolated)"])

# ══════════════════════════════════════════════════════════════
# 6) Energy × Digital 4분류
# ══════════════════════════════════════════════════════════════
joined["energy_access"]  = joined["ElecStart"] == 1
joined["digital_access"] = joined["map_conn_status"] == "Connected"

def typology(r):
    if r["energy_access"] and r["digital_access"]: return "Energy + Digital"
    if r["energy_access"]:                          return "Energy only"
    if r["digital_access"]:                         return "Digital only"
    return "No energy or digital"

joined["combined_access"] = joined.apply(typology, axis=1)

# ══════════════════════════════════════════════════════════════
# 7) 저장 · 요약
# ══════════════════════════════════════════════════════════════
out_csv = os.path.join(OUT_DIR, "schools_TZA_energy_digital.csv")
joined.drop(columns="geometry").to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\n=== 매칭 방식 ===")
print(joined["match_type"].value_counts())
print("\n=== 신뢰도 ===")
print(joined["energy_confidence"].value_counts())
print("\n=== 4분류 ===")
print(joined["combined_access"].value_counts())
print("\n=== 교차표 ===")
print(pd.crosstab(joined["energy_access"], joined["digital_access"],
                  rownames=["Energy"], colnames=["Digital"]))
if "elec_rate" in joined.columns:
    print("\n분류별 평균 전기화율:")
    print(joined.groupby("combined_access")["elec_rate"].mean().round(3))
print(f"\n저장 → {out_csv}")