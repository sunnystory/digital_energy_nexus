import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════
# 1) 국가별 설정
# ══════════════════════════════════════════════════════════════
BASE = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access"

COUNTRY_CFG = {
    "TZA": dict(name="Tanzania",
                gep_dir="tz-3-scenarios-results",
                clusters=r"tz-1\final_clusters.shp",
                metric_crs="EPSG:32736"),
    "MWI": dict(name="Malawi",
                gep_dir="mw-3-scenarios-results",
                clusters=r"mw-1\final_clusters.shp",
                metric_crs="EPSG:32736"),
    "LSO": dict(name="Lesotho",
                gep_dir="ls-3-scenarios-results",
                clusters=r"ls-1\final_clusters.shp",
                metric_crs="EPSG:32735"),
    "BWA": dict(name="Botswana",
                gep_dir="bw-3-scenarios-results",
                clusters=r"bw-1\final_clusters.shp",
                metric_crs="EPSG:32734"),
}

SCHOOL_DIR = os.path.join(BASE, "merged")
OUT_DIR    = os.path.join(BASE, "energy_digital")
MAP_DIR    = os.path.join(OUT_DIR, "maps")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

NEAR_CLUSTER_MAX_M = 2000

COLORS = {
    "Energy + Digital":     "#2166ac",   # 진한 파랑
    "Energy only":          "#67a9cf",   # 연한 파랑
    "Digital only":         "#ef8a62",   # 주황
    "No energy or digital": "#b2182b",   # 빨강
}
ORDER = ["Energy + Digital", "Energy only", "Digital only", "No energy or digital"]

TECH = {1:"Existing grid", 2:"Grid extension", 3:"Solar MG",
        5:"Hydro MG", 6:"Wind MG", 7:"Standalone PV", 99:"Not electrified"}

# ══════════════════════════════════════════════════════════════
# 2) 한 국가 처리 함수
# ══════════════════════════════════════════════════════════════
def process_country(iso3, cfg):
    print(f"\n{'='*60}\n{cfg['name']} ({iso3})\n{'='*60}")

    gep_dir   = os.path.join(BASE, cfg["gep_dir"])
    clus_path = os.path.join(BASE, cfg["clusters"])
    sch_path  = os.path.join(SCHOOL_DIR, f"schools_{iso3}.csv")

    # 파일 존재 확인
    for p, label in [(gep_dir, "GEP 폴더"), (clus_path, "클러스터 shp"), (sch_path, "학교 csv")]:
        if not os.path.exists(p):
            print(f"  ⚠ {label} 없음 → 건너뜀: {p}")
            return None

    # ── GEP: 시나리오 파일 아무거나 하나 (2020 컬럼은 모두 동일) ──
    csvs = sorted([f for f in os.listdir(gep_dir) if f.endswith(".csv")])
    if not csvs:
        print(f"  ⚠ GEP csv 없음"); return None
    gep = pd.read_csv(os.path.join(gep_dir, csvs[0]), low_memory=False)
    print(f"  GEP: {csvs[0]}  {gep.shape}")

    keep = ["id","ElecStart","FinalElecCode2020","ElecPopCalib","Pop2020",
            "PopStartYear","NightLights","IsUrban","CurrentMVLineDist",
            "TransformerDist","GridDistCalibElec","Prim","Sec","Unc","Admin1"]
    gep = gep[[c for c in keep if c in gep.columns]].copy()

    pop_col = "Pop2020" if "Pop2020" in gep.columns else "PopStartYear"
    if "ElecPopCalib" in gep.columns and pop_col in gep.columns:
        gep["elec_rate"] = (gep["ElecPopCalib"]/gep[pop_col].replace(0, pd.NA)).clip(0,1)
    if "FinalElecCode2020" in gep.columns:
        gep["elec_tech_2020"] = gep["FinalElecCode2020"].map(TECH)

    # ── 클러스터 ──
    clusters = gpd.read_file(clus_path)
    id_col = "id" if "id" in clusters.columns else clusters.columns[0]
    clusters = clusters[[id_col,"geometry"]].merge(gep, left_on=id_col, right_on="id", how="left")
    if clusters.crs is None:
        clusters = clusters.set_crs("EPSG:4326")
    clusters = clusters.to_crs("EPSG:4326")
    ENERGY_COLS = [c for c in clusters.columns if c != "geometry"]
    print(f"  클러스터: {len(clusters)} | GEP 결합: {clusters['ElecStart'].notna().mean():.1%}")

    # ── 학교 ──
    sch = pd.read_csv(sch_path, low_memory=False).dropna(subset=["latitude","longitude"])
    sch_gdf = gpd.GeoDataFrame(sch,
        geometry=gpd.points_from_xy(sch["longitude"], sch["latitude"]), crs="EPSG:4326")
    print(f"  학교: {len(sch_gdf)}")

    # ── 공간조인 + 최근접 보완 ──
    j = gpd.sjoin(sch_gdf, clusters, how="left", predicate="within")
    j = j.drop(columns=[c for c in ["index_right"] if c in j.columns])
    j = j.drop_duplicates(subset="giga_id_school", keep="first")
    print(f"  within 매칭: {j['ElecStart'].notna().mean():.1%}")

    j["match_type"] = "within"; j["cluster_dist_m"] = 0.0
    miss = j.loc[j["ElecStart"].isna(), "giga_id_school"]

    if len(miss) > 0:
        pts = sch_gdf[sch_gdf["giga_id_school"].isin(miss)].to_crs(cfg["metric_crs"])
        near = gpd.sjoin_nearest(pts, clusters.to_crs(cfg["metric_crs"]),
                                 how="left", distance_col="cluster_dist_m")
        near = near.drop(columns=[c for c in ["index_right"] if c in near.columns])
        near = near.drop_duplicates(subset="giga_id_school", keep="first")
        print(f"  폴리곤 밖 {len(near)}개 | 중앙값 {near['cluster_dist_m'].median():.0f}m")

        j = j.set_index("giga_id_school"); near = near.set_index("giga_id_school")
        close = near[near["cluster_dist_m"] <= NEAR_CLUSTER_MAX_M]
        far   = near[near["cluster_dist_m"] >  NEAR_CLUSTER_MAX_M]

        for c in ENERGY_COLS:
            if c in close.columns:
                j.loc[close.index, c] = close[c]
        j.loc[close.index, ["cluster_dist_m"]] = close[["cluster_dist_m"]]
        j.loc[close.index, "match_type"] = "nearest"

        j.loc[far.index, "ElecStart"] = 0
        j.loc[far.index, "elec_rate"] = 0.0
        j.loc[far.index, "elec_tech_2020"] = "Not electrified (isolated)"
        j.loc[far.index, ["cluster_dist_m"]] = far[["cluster_dist_m"]]
        j.loc[far.index, "match_type"] = "isolated_no_elec"
        j = j.reset_index()
        print(f"  보완 {len(close)}개 / 외딴 {len(far)}개")

    # ── 4분류 ──
    j["energy_access"]  = j["ElecStart"] == 1
    j["digital_access"] = j["map_conn_status"] == "Connected"
    j["combined_access"] = [
        "Energy + Digital" if e and d else
        "Energy only"      if e else
        "Digital only"     if d else
        "No energy or digital"
        for e, d in zip(j["energy_access"], j["digital_access"])
    ]
    j["country_iso3"] = iso3
    j["country"]      = cfg["name"]

    # ── 저장 ──
    out = os.path.join(OUT_DIR, f"schools_{iso3}_energy_digital.csv")
    j.drop(columns="geometry").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  저장 → {out}")
    print(j["combined_access"].value_counts().to_string())

    return j

# ══════════════════════════════════════════════════════════════
# 3) 정적 지도 (matplotlib)
# ══════════════════════════════════════════════════════════════
def make_map(gdf, iso3, name):
    fig, ax = plt.subplots(figsize=(9, 10), dpi=200)

    # 뒤에 깔리는 순서: 많은 것부터 → 적은 것 위로
    for cat in ["No energy or digital", "Energy only", "Digital only", "Energy + Digital"]:
        sub = gdf[gdf["combined_access"] == cat]
        if sub.empty: continue
        ax.scatter(sub.geometry.x, sub.geometry.y,
                   s=2.5, c=COLORS[cat], alpha=0.65, linewidths=0, label=cat)

    total = len(gdf)
    handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS[c],
                      markersize=8,
                      label=f"{c}  {(gdf['combined_access']==c).sum():,} "
                            f"({(gdf['combined_access']==c).mean()*100:.1f}%)")
               for c in ORDER if (gdf["combined_access"]==c).any()]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              frameon=True, framealpha=0.9, title="Access typology", title_fontsize=9)

    ax.set_title(f"{name} — School Energy & Digital Access\n"
                 f"n = {total:,} schools", fontsize=13, pad=12)
    ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.2, linewidth=0.4)
    ax.tick_params(labelsize=8)

    path = os.path.join(MAP_DIR, f"map_{iso3}_access_typology.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  지도 → {path}")

# ══════════════════════════════════════════════════════════════
# 4) 전체 실행
# ══════════════════════════════════════════════════════════════
results = {}
for iso3, cfg in COUNTRY_CFG.items():
    g = process_country(iso3, cfg)
    if g is not None:
        results[iso3] = g
        make_map(g, iso3, cfg["name"])

if not results:
    raise SystemExit("처리된 국가가 없습니다.")

# ══════════════════════════════════════════════════════════════
# 5) 요약표
# ══════════════════════════════════════════════════════════════
all_g = pd.concat([g.drop(columns="geometry") for g in results.values()], ignore_index=True)

# (a) 국가 × 4분류 — 건수
tbl_n = pd.crosstab(all_g["country"], all_g["combined_access"])
tbl_n = tbl_n.reindex(columns=[c for c in ORDER if c in tbl_n.columns])
tbl_n["Total"] = tbl_n.sum(axis=1)

# (b) 국가 × 4분류 — 비율(%)
tbl_p = (tbl_n.drop(columns="Total")
         .div(tbl_n["Total"], axis=0).mul(100).round(1))

# (c) 국가별 핵심 지표
summary = pd.DataFrame({
    "Schools":        all_g.groupby("country").size(),
    "Energy access %":  all_g.groupby("country")["energy_access"].mean().mul(100).round(1),
    "Digital access %": all_g.groupby("country")["digital_access"].mean().mul(100).round(1),
    "Both %":  all_g.groupby("country").apply(
        lambda d: (d["combined_access"]=="Energy + Digital").mean()*100).round(1),
    "Neither %": all_g.groupby("country").apply(
        lambda d: (d["combined_access"]=="No energy or digital").mean()*100).round(1),
    "Mean elec_rate": all_g.groupby("country")["elec_rate"].mean().round(3),
})

# (d) 분류별 평균 전기화율
tbl_rate = all_g.pivot_table(index="country", columns="combined_access",
                             values="elec_rate", aggfunc="mean").round(3)
tbl_rate = tbl_rate.reindex(columns=[c for c in ORDER if c in tbl_rate.columns])

print(f"\n{'='*60}\n요약\n{'='*60}")
print("\n[국가 × 분류 — 건수]\n", tbl_n.to_string())
print("\n[국가 × 분류 — %]\n", tbl_p.to_string())
print("\n[국가별 핵심 지표]\n", summary.to_string())
print("\n[분류별 평균 전기화율]\n", tbl_rate.to_string())

# 엑셀로 저장 (시트 분리)
xlsx = os.path.join(OUT_DIR, "summary_energy_digital.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    tbl_n.to_excel(w, sheet_name="counts")
    tbl_p.to_excel(w, sheet_name="percent")
    summary.to_excel(w, sheet_name="key_indicators")
    tbl_rate.to_excel(w, sheet_name="elec_rate")
all_g.to_csv(os.path.join(OUT_DIR, "schools_all_countries.csv"),
             index=False, encoding="utf-8-sig")
print(f"\n저장 → {xlsx}")