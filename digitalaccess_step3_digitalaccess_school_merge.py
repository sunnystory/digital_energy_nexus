import os
import pandas as pd

# ══════════════════════════════════════════════════════════════
# 1) 경로 설정
# ══════════════════════════════════════════════════════════════
LOC_DIR  = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\school"
PROF_PATH = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\giga.csv"
OUT_DIR   = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\merged"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES = {
    "BWA": "Botswana",
    "LSO": "Lesotho",
    "MWI": "Malawi",
    "TZA": "United Republic of Tanzania",
}

# ══════════════════════════════════════════════════════════════
# 2) profile 연결정보 준비 (한 번만 불러옴)
# ══════════════════════════════════════════════════════════════
prof = pd.read_csv(PROF_PATH, index_col=0, low_memory=False)

def conn_status(row):
    t = str(row["connectivity_type"]).strip().lower()
    type_known = t not in ("unknown", "none", "nan", "")
    if row["connectivity_rt"] == True or type_known:
        return "Connected"
    return "Not Connected / Unknown"

prof["map_conn_status"] = prof.apply(conn_status, axis=1)
prof_cols = ["giga_id_school", "connectivity_rt", "connectivity_type", "map_conn_status"]

# ══════════════════════════════════════════════════════════════
# 3) 국가별로: location 불러오기 → merge → 개별 CSV 저장
# ══════════════════════════════════════════════════════════════
for iso3, country in COUNTRIES.items():
    loc_path = os.path.join(LOC_DIR, f"location_{iso3}.csv")
    if not os.path.exists(loc_path):
        print(f"⚠ 파일 없음: {loc_path}"); continue

    loc = pd.read_csv(loc_path)
    loc = loc.dropna(subset=["latitude", "longitude"])   # 좌표 없는 행 제거

    merged = loc.merge(prof[prof_cols], on="giga_id_school", how="left")
    merged["map_conn_status"] = merged["map_conn_status"].fillna("No profile data")

    out_path = os.path.join(OUT_DIR, f"schools_{iso3}.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 요약 출력
    total = len(merged)
    connected = (merged["map_conn_status"] == "Connected").sum()
    print(f"\n{country} ({iso3})")
    print(f"  학교 수: {total}")
    print(f"  Connected: {connected}  |  나머지: {total - connected}")
    print(f"  저장 → {out_path}")

print("\n완료!")