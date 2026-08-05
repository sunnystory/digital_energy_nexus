import requests, time, os
import pandas as pd

# ══════════════════════════════════════════════════════════════
# 1) 설정
# ══════════════════════════════════════════════════════════════
API_KEY = ""   # ← 직접 입력

BASE = "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/schools_location"
PAGE_SIZE, TIMEOUT, MAX_RETRIES = 1000, 60, 3
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

COUNTRIES = {
    "BWA": "Botswana",
    "LSO": "Lesotho",
    "MWI": "Malawi",
    "TZA": "United Republic of Tanzania",
}

OUT_DIR   = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\school"
PROF_PATH = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\giga.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2) 응답 파싱 + 요청 (문서: 국가별은 /country/{iso3} 경로)
# ══════════════════════════════════════════════════════════════
def extract_records(payload):
    if isinstance(payload, dict):
        recs = payload.get("data")
        if not isinstance(recs, list):
            lists = [v for v in payload.values() if isinstance(v, list)]
            recs = lists[0] if lists else []
    elif isinstance(payload, list):
        recs = payload
    else:
        recs = []
    return [r for r in recs if isinstance(r, dict)]

def get_json(url, params):
    """재시도 포함 GET. (records, ok) 반환."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"    network error (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(2 * attempt); continue
        if r.status_code == 200:
            return extract_records(r.json()), True
        if r.status_code == 404:
            return [], True
        if r.status_code in (401, 403):
            print(f"    auth error {r.status_code}: check API_KEY"); return [], False
        if 500 <= r.status_code < 600:
            print(f"    server error {r.status_code} (attempt {attempt}); retrying")
            time.sleep(2 * attempt); continue
        print(f"    unexpected {r.status_code}: {r.text[:200]}"); return [], False
    print("    gave up after retries"); return [], False

def download_country(iso3, country):
    """국가별 경로로 location 전체를 받는다.
       페이징을 시도하되, 한 번에 다 오는 응답도 처리."""
    print(f"\nLocation 다운로드: {country} ({iso3})")
    url = f"{BASE}/country/{iso3}"
    rows, page, total = [], 1, 0
    while True:
        data, ok = get_json(url, params={"page": page, "size": PAGE_SIZE})
        if not ok:
            break
        if len(data) == 0:
            break
        for row in data:
            row["country_name"] = country
        rows.extend(data); total += len(data)
        print(f"  page {page}: {len(data)} records")
        if len(data) < PAGE_SIZE:   # 마지막(부분) 페이지 → 종료
            break
        page += 1; time.sleep(0.2)
    print(f"  Total {country}: {total}")
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# 3) 국가별로 받아서 각각 CSV 저장
# ══════════════════════════════════════════════════════════════
location_frames = []
for iso3, country in COUNTRIES.items():
    df_loc = download_country(iso3, country)
    if df_loc.empty:
        print(f"  ⚠ {country}: 데이터 없음, 건너뜀"); continue
    path = os.path.join(OUT_DIR, f"location_{iso3}.csv")
    df_loc.to_csv(path, index=False, encoding="utf-8-sig")  # 한글/특수문자 안전
    print(f"  저장됨 → {path}  {df_loc.shape}")
    location_frames.append(df_loc)

if not location_frames:
    raise SystemExit("받아온 location 데이터가 없습니다.")

print("\n=== location 컬럼 목록 ===")
print(list(location_frames[0].columns))

# ══════════════════════════════════════════════════════════════
# 4) location 합치기
# ══════════════════════════════════════════════════════════════
loc_all = pd.concat(location_frames, ignore_index=True)
print("\nlocation 합계:", loc_all.shape)

# 좌표 결측/이상치 간단 점검
bad = loc_all[["latitude", "longitude"]].isna().any(axis=1).sum()
print(f"좌표 결측 행: {bad}")

# ══════════════════════════════════════════════════════════════
# 5) profile(연결정보) 준비 — 대상국만 필터 + 상태 컬럼
# ══════════════════════════════════════════════════════════════
prof = pd.read_csv(PROF_PATH, index_col=0, low_memory=False)
prof = prof[prof["country_name"].isin(COUNTRIES.values())].copy()

def conn_status(row):
    t = str(row["connectivity_type"]).strip().lower()
    type_known = t not in ("unknown", "none", "nan", "")
    if row["connectivity_rt"] == True or type_known:
        return "Connected"
    return "Not Connected / Unknown"

prof["map_conn_status"] = prof.apply(conn_status, axis=1)

# ══════════════════════════════════════════════════════════════
# 6) 위치 + 연결정보 merge (giga_id_school 기준)
# ══════════════════════════════════════════════════════════════
prof_cols = ["giga_id_school", "connectivity_rt", "connectivity_type", "map_conn_status"]
merged = loc_all.merge(prof[prof_cols], on="giga_id_school", how="left")

out_path = r"C:\Users\wb632724\Downloads\e&d\giga_energy_access\schools_location_connectivity_4countries.csv"
merged.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n=== merge 결과 ===")
print("shape:", merged.shape)
print("연결정보 붙은 비율:", f"{merged['map_conn_status'].notna().mean():.1%}")
print()
print(merged.groupby("country_name")["map_conn_status"].value_counts(dropna=False))