import json
import geopandas as gpd
from shapely.ops import unary_union

# 1) admdongkor의 서울 GeoJSON 다운로드 후 로컬에 저장해 둔 파일 경로
#    예: data/seoul_admdong.geojson
INPUT = "data/seoul_admdong.geojson"

# 2) 결과 파일
OUTPUT = "data/jongno_boundary.geojson"

TARGET_GU_FULL = "서울특별시 종로구"

gdf = gpd.read_file(INPUT)

# 보통 컬럼명이 adm_nm (예: "서울특별시 종로구 사직동") 형태
if "adm_nm" not in gdf.columns:
    raise ValueError(f"'adm_nm' 컬럼이 없습니다. 현재 컬럼: {list(gdf.columns)}")

jongno = gdf[gdf["adm_nm"].str.startswith(TARGET_GU_FULL)].copy()
if jongno.empty:
    raise ValueError("종로구 데이터를 못 찾았습니다. adm_nm 값 예시를 확인하세요.")

# 행정동 폴리곤을 종로구 1개 폴리곤으로 합치기
geom = unary_union(jongno.geometry)

out = gpd.GeoDataFrame({"name": ["종로구"]}, geometry=[geom], crs=gdf.crs)
out = out.to_crs(epsg=4326)  # WGS84로 고정(지도에 바로 쓰기 좋음)

out.to_file(OUTPUT, driver="GeoJSON")
print("Saved:", OUTPUT)
