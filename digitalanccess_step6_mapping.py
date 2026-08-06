import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
BASE      = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access"
OUT_DIR   = os.path.join(BASE, "energy_digital")
MAP_DIR   = os.path.join(OUT_DIR, "maps")
BOUND_DIR = os.path.join(BASE, "boundaries")
os.makedirs(MAP_DIR, exist_ok=True)

COUNTRIES = {"TZA": "Tanzania", "MWI": "Malawi",
             "LSO": "Lesotho",  "BWA": "Botswana"}

ADM_LEVEL = 1        # 1=주/지역, 2=군/구, 3=읍면
LABEL_ADM = False    # 행정구역 이름 표시 여부

COLORS = {
    "Energy + Digital":     "#2166ac",   # 진한 파랑
    "Energy only":          "#67a9cf",   # 연한 파랑
    "Digital only":         "#ef8a62",   # 주황
    "No energy or digital": "#b2182b",   # 빨강
}
ORDER      = ["Energy + Digital", "Energy only", "Digital only", "No energy or digital"]
DRAW_ORDER = ["No energy or digital", "Energy only", "Digital only", "Energy + Digital"]

# ══════════════════════════════════════════════════════════════
# 1) 경계 불러오기 (하위 폴더까지 재귀 탐색)
# ══════════════════════════════════════════════════════════════
def load_boundary(iso3, level=1):
    """boundaries/ 및 그 하위 폴더에서 GADM 경계를 찾는다.
       예: boundaries/gadm41_TZA_shp/gadm41_TZA_1.shp"""
    adm, ctry = None, None

    def find(patterns):
        for pat in patterns:
            hits = glob.glob(os.path.join(BOUND_DIR, "**", pat), recursive=True)
            if hits:
                return sorted(hits)[0]
        return None

    p_adm = find([f"gadm41_{iso3}_{level}.shp", f"gadm41_{iso3}_{level}.gpkg",
                  f"{iso3}_adm{level}.shp", f"{iso3}_ADM{level}.shp"])
    if p_adm:
        adm = gpd.read_file(p_adm).to_crs("EPSG:4326")
        print(f"    경계 ADM{level}: {os.path.basename(p_adm)}  ({len(adm)} features)")

    p_ctry = find([f"gadm41_{iso3}_0.shp", f"gadm41_{iso3}_0.gpkg", f"{iso3}_adm0.shp"])
    if p_ctry:
        ctry = gpd.read_file(p_ctry).to_crs("EPSG:4326")
        print(f"    경계 ADM0: {os.path.basename(p_ctry)}")

    if adm is None and ctry is None:
        print(f"    ⚠ {iso3} 경계 파일 없음 (검색 위치: {BOUND_DIR})")

    return adm, ctry

# ══════════════════════════════════════════════════════════════
# 2) 지도 그리기
# ══════════════════════════════════════════════════════════════
def make_map(gdf, iso3, name, adm_level=1, label_adm=False):
    adm, ctry = load_boundary(iso3, level=adm_level)

    fig, ax = plt.subplots(figsize=(9, 10), dpi=200)

    # ── 배경: 국가 면 → 행정구역 경계선 ──
    if ctry is not None and len(ctry):
        ctry.plot(ax=ax, color="#f4f4f4", edgecolor="#4d4d4d",
                  linewidth=1.1, zorder=0)

    if adm is not None and len(adm):
        adm.plot(ax=ax, facecolor="none", edgecolor="#a8a8a8",
                 linewidth=0.5, zorder=1)

        if label_adm:
            namecol = next((c for c in ["NAME_1", "NAME_2", "NAME_3", "ADM1_EN"]
                            if c in adm.columns), None)
            if namecol:
                for _, r in adm.iterrows():
                    pt = r.geometry.representative_point()
                    ax.annotate(r[namecol], (pt.x, pt.y), fontsize=5.5,
                                color="#555555", ha="center", zorder=2)

    # ── 학교 점 (많은 분류부터 아래에) ──
    for cat in DRAW_ORDER:
        sub = gdf[gdf["combined_access"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub.geometry.x, sub.geometry.y, s=2.5, c=COLORS[cat],
                   alpha=0.7, linewidths=0, zorder=3)

    # ── 범례 (건수·비율 자동 표기) ──
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[c],
                      markersize=8,
                      label=f"{c}  {(gdf['combined_access']==c).sum():,} "
                            f"({(gdf['combined_access']==c).mean()*100:.1f}%)")
               for c in ORDER if (gdf["combined_access"] == c).any()]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True,
              framealpha=0.92, title="Access typology", title_fontsize=9)

    # ── 범위: 경계와 학교를 모두 포함 ──
    bounds = gdf.total_bounds
    if ctry is not None and len(ctry):
        cb = ctry.total_bounds
        bounds = [min(bounds[0], cb[0]), min(bounds[1], cb[1]),
                  max(bounds[2], cb[2]), max(bounds[3], cb[3])]
    minx, miny, maxx, maxy = bounds
    mx, my = (maxx-minx)*0.03, (maxy-miny)*0.03
    ax.set_xlim(minx-mx, maxx+mx); ax.set_ylim(miny-my, maxy+my)

    ax.set_title(f"{name} — School Energy & Digital Access\nN = {len(gdf):,} schools",
                 fontsize=13, pad=12)
    ax.set_aspect("equal")
    ax.set_axis_off()          # 축·눈금·테두리·그리드 모두 제거

    path = os.path.join(MAP_DIR, f"map_{iso3}_access_typology.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"    지도 → {path}")

# ══════════════════════════════════════════════════════════════
# 3) 실행: 저장된 결과 CSV로 지도 생성
# ══════════════════════════════════════════════════════════════
for iso3, name in COUNTRIES.items():
    p = os.path.join(OUT_DIR, f"schools_{iso3}_energy_digital.csv")
    if not os.path.exists(p):
        print(f"\n{name} ({iso3}): 결과 파일 없음, 건너뜀")
        continue

    print(f"\n{name} ({iso3})")
    df = pd.read_csv(p, low_memory=False).dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326")
    make_map(gdf, iso3, name, adm_level=ADM_LEVEL, label_adm=LABEL_ADM)

print("\n완료")