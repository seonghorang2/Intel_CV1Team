import os
from datetime import datetime, timedelta
import json
import hashlib

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from shapely.geometry import shape, Point

# ✅ db.py 그대로 사용
from db import init_db, insert_event, fetch_events

# =========================
# Page Config (사이드바는 유지, 기본은 접힘)
# =========================
st.set_page_config(
    page_title="고령자 낙상 예방 관제 시스템",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ADMDONG_PATH = os.path.join("data", "seoul_admdong.geojson")


TARGET_GU = "종로구"
CCTV_CSV_PATH = os.path.join("data", "seoul_cctv.csv")
JONGNO_BOUNDARY_PATH = os.path.join("data", "jongno_boundary.geojson")

# ✅ 노트북(웹캠) 고정 좌표(데이터 보강용)
LAPTOP_LAT = 37.583266
LAPTOP_LON = 126.966548
SOURCE_ID = "laptop_cam_01"

LAPTOP_ROW = {
    "자치구": TARGET_GU,
    "안심 주소": "서울특별시 종로구 옥인동 47-264(노트북 웹캠)",
    "위도": LAPTOP_LAT,
    "경도": LAPTOP_LON,
    "CCTV 수량": 1,
    "수정 일시": "",
}

# =========================
# Sidebar - Controls (왼쪽 팝업 그대로)
# =========================
st.sidebar.header("⚙️ 화면 설정")

show_environment = st.sidebar.checkbox("환경 정보 표시(지도 아래)", value=True)

st.sidebar.divider()
st.sidebar.subheader("🗺️ CCTV 점 표시 (지도=좌표 통합)")
show_all_points = st.sidebar.checkbox("전체 CCTV 위치 표시", value=False)
show_high_points = st.sidebar.checkbox("High 위치 표시", value=True)
show_medium_points = st.sidebar.checkbox("Medium 위치 표시", value=True)

st.sidebar.divider()
show_event_hex = st.sidebar.checkbox("이벤트 격자(HEX) 표시", value=False)

st.sidebar.divider()
time_window = st.sidebar.radio("누적 기준", ["최근 4시간", "최근 24시간", "최근 72시간"], index=1)

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("자동 새로고침 사용", value=False)
refresh_minutes = st.sidebar.selectbox("새로고침 주기(분)", [1, 5, 10, 30], index=1)

if st.sidebar.button("지금 새로고침"):
    st.rerun()

if auto_refresh:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# =========================
# Helpers
# =========================
@st.cache_data
def build_dong_polygons(admdong_jongno_fc: dict):
    """
    return: list of (dong_name, shapely_polygon)
    """
    polys = []
    for f in admdong_jongno_fc.get("features", []):
        props = f.get("properties", {})
        adm_nm = str(props.get("adm_nm", "")).strip()  # "서울특별시 종로구 사직동"
        dong = adm_nm.split()[-1] if adm_nm else "미상"
        geom = f.get("geometry")
        if not geom:
            continue
        polys.append((dong, shape(geom)))
    return polys


def assign_dong_nearest(lat: float, lon: float, dong_polys) -> str:
    """
    폴리곤 내부면 그 동, 아니면 가장 가까운 동(거리 최소)
    """
    p = Point(lon, lat)

    # 1) contains 우선
    for dong, poly in dong_polys:
        if poly.contains(p):
            return dong

    # 2) 미분류 -> nearest polygon (boundary distance)
    best_dong = "미분류"
    best_dist = float("inf")
    for dong, poly in dong_polys:
        d = poly.distance(p)  # degree 단위지만 "가까운 동 선택"에는 충분
        if d < best_dist:
            best_dist = d
            best_dong = dong
    return best_dong

def to_kst(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("Asia/Seoul")

def filter_admdong_by_gu(geojson_fc: dict, target_gu: str) -> dict:
    kept = []
    for f in geojson_fc.get("features", []):
        props = f.get("properties", {})
        adm_nm = str(props.get("adm_nm", "")).strip()  # ✅ "서울특별시 종로구 사직동"
        if target_gu in adm_nm:  # ✅ "종로구" 포함이면 종로구 동
            kept.append(f)

    return {
        "type": "FeatureCollection",
        "features": kept,
    }


def layer_dong_outline(geojson_fc: dict):
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_fc,
        stroked=True,
        filled=False,
        get_line_color=[80, 80, 80, 90],  # 연회색 + 투명
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=2,
        pickable=False,
    )

def _flatten_coords(coords):
    """
    Polygon/MultiPolygon 좌표를 재귀적으로 펼쳐서 (lon, lat) 리스트로 반환
    """
    pts = []

    def walk(c):
        if isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[0], (int, float)):
            lon, lat = c
            pts.append((lon, lat))
        else:
            for x in c:
                walk(x)

    walk(coords)
    return pts

def make_dong_label_points(admdong_fc: dict) -> pd.DataFrame:
    rows = []
    for f in admdong_fc.get("features", []):
        props = f.get("properties", {})
        name = str(props.get("adm_nm", "")).strip()  # "서울특별시 종로구 사직동"

        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [])

        pts = _flatten_coords(coords)
        if not pts:
            continue

        lon_avg = sum(p[0] for p in pts) / len(pts)
        lat_avg = sum(p[1] for p in pts) / len(pts)

        short = name.split()[-1] if name else ""
        rows.append({"adm_nm": name, "dong": short, "lon": lon_avg, "lat": lat_avg})

    return pd.DataFrame(rows)


def layer_dong_labels(df_labels: pd.DataFrame):
    if df_labels is None or df_labels.empty:
        return None

    df = df_labels.copy()

    # ✅ label 컬럼 강제 생성 (dong가 없으면 adm_nm 마지막 단어로라도)
    if "dong" in df.columns:
        df["label"] = df["dong"].astype(str)
    elif "adm_nm" in df.columns:
        df["label"] = df["adm_nm"].astype(str).apply(lambda x: str(x).split()[-1])
    else:
        return None

    # ✅ 좌표 타입 보정
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "label"])
    if df.empty:
        return None

    return pdk.Layer(
        "TextLayer",
        data=df,
        get_position="[lon, lat]",
        get_text="label",
        get_size=16,
        size_units="pixels",
        size_min_pixels=12,
        size_max_pixels=28,
        billboard=True,              # ✅ 카메라 회전에 상관없이 정면 표시
        get_color=[20, 20, 20, 230],
        pickable=False,
    )




def make_last_4hour_bins_kst():
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").floor("H")
    hours_kst = [now_kst - pd.Timedelta(hours=i) for i in range(3, -1, -1)]
    idx_kst = pd.DatetimeIndex(hours_kst)

    labels = [
        f"{h.strftime('%H:%M')}~{(h + pd.Timedelta(hours=1)).strftime('%H:%M')}"
        for h in idx_kst
    ]
    return idx_kst, labels


def get_window_hours() -> int:
    return {"최근 4시간": 4, "최근 24시간": 24, "최근 72시간": 72}[time_window]


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="euc-kr")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


@st.cache_data
def load_boundary_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_coords(coords):
    # Polygon/MultiPolygon 좌표를 전부 순회하면서 (lon, lat) 뽑기
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        yield coords  # [lon, lat]
    else:
        for c in coords:
            yield from _iter_coords(c)

def geojson_centroid_lonlat(geom: dict):
    # 정확한 중심(지오메트리 중심점) 계산은 shapely가 필요하지만,
    # 여기선 "라벨 위치용"으로 bbox 중심을 사용 (충분히 보기 좋음)
    coords = list(_iter_coords(geom.get("coordinates", [])))
    if not coords:
        return None, None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2

@st.cache_data
def build_dong_label_df(seoul_dong_geojson_path: str, target_gu: str) -> pd.DataFrame:
    with open(seoul_dong_geojson_path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    rows = []
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        adm_nm = str(props.get("adm_nm", ""))  # 예: "서울특별시 종로구 사직동"
        key = f" {target_gu} "
        if key not in adm_nm:
            continue

        # 동 이름만 뽑기: "서울특별시 종로구 사직동" -> "사직동"
        dong_name = adm_nm.split()[-1]

        lon, lat = geojson_centroid_lonlat(feat.get("geometry", {}))
        if lon is None:
            continue

        rows.append({"dong": dong_name, "lon": lon, "lat": lat})

    return pd.DataFrame(rows)


def layer_gu_outline(geojson_fc: dict):
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_fc,
        stroked=True,
        filled=False,
        get_line_color=[0, 120, 255, 110],
        get_line_width=2,
        line_width_min_pixels=1,
        line_width_max_pixels=3,
        pickable=False,
    )


def make_camera_id(address: str, lat: float, lon: float) -> str:
    s = f"{address}|{lat:.6f}|{lon:.6f}".encode("utf-8")
    h = hashlib.sha1(s).hexdigest()[:10]
    return f"CAM_{h}"


@st.cache_data
def load_cctv_data():
    """
    cameras_df: 주소 단위(중복 좌표 허용)
    sites_df: 좌표 단위(지도용 통합)
    """
    if not os.path.exists(CCTV_CSV_PATH):
        raise FileNotFoundError(
            f"CSV를 찾을 수 없습니다: {CCTV_CSV_PATH}\n"
            f"project_1/data/seoul_cctv.csv 경로로 넣어주세요."
        )

    df = safe_read_csv(CCTV_CSV_PATH)

    required = ["자치구", "안심 주소", "위도", "경도", "CCTV 수량", "수정 일시"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 컬럼이 예상과 다릅니다. 누락: {missing}\n현재 컬럼: {list(df.columns)}")

    df = df[df["자치구"] == TARGET_GU].copy()

    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    df["CCTV 수량"] = pd.to_numeric(df["CCTV 수량"], errors="coerce").fillna(1).astype(int)
    df = df.dropna(subset=["위도", "경도", "안심 주소"]).copy()

    # 노트북(웹캠) 추가
    df = pd.concat([df, pd.DataFrame([LAPTOP_ROW])], ignore_index=True)

    cameras = df.copy()
    cameras["lat"] = cameras["위도"].astype(float)
    cameras["lon"] = cameras["경도"].astype(float)
    cameras["camera_id"] = cameras.apply(
        lambda r: make_camera_id(str(r["안심 주소"]), float(r["lat"]), float(r["lon"])),
        axis=1
    )
    cameras = cameras.drop_duplicates(subset=["camera_id"]).reset_index(drop=True)

    sites = (
        cameras.groupby(["lat", "lon"], as_index=False)
        .agg({"자치구": "first", "CCTV 수량": "sum", "안심 주소": "count"})
        .rename(columns={"안심 주소": "카메라 수"})
        .copy()
    )
    sites["site_id"] = "SITE_" + sites.index.astype(str).str.zfill(5)

    cameras = cameras.merge(sites[["lat", "lon", "site_id"]], on=["lat", "lon"], how="left")
    return cameras, sites


def load_events_df(limit: int = 8000) -> pd.DataFrame:
    rows = fetch_events(limit=limit)
    if not rows:
        return pd.DataFrame(columns=["ts", "lat", "lon", "dong", "cctv_id", "event_type", "confidence", "source_id"])

    df = pd.DataFrame(rows, columns=["ts", "lat", "lon", "dong", "cctv_id", "event_type", "confidence", "source_id"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["ts", "lat", "lon", "cctv_id"]).copy()
    return df


def filter_events_by_time(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(hours=get_window_hours())
    return df[df["ts"] >= pd.Timestamp(cutoff)].copy()


def priority_from_count(n: int) -> str:
    if n >= 3:
        return "High"
    elif n >= 1:
        return "Medium"
    return "Low"


def scatter_layer(df: pd.DataFrame, radius: int, color_rgba: list):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=radius,
        get_fill_color=color_rgba,
        pickable=True,
    )


def layer_event_hex(df_events: pd.DataFrame):
    if df_events.empty:
        return None
    return pdk.Layer(
        "HexagonLayer",
        data=df_events,
        get_position="[lon, lat]",
        radius=35,
        elevation_scale=0,
        extruded=False,
        pickable=True,
    )
    
def point_in_polygon(lon: float, lat: float, ring: list) -> bool:
    """
    ring: [[lon,lat], [lon,lat], ...]  (Polygon의 바깥쪽 링 1개)
    Ray casting 알고리즘
    """
    inside = False
    n = len(ring)
    if n < 3:
        return False

    x, y = lon, lat
    x0, y0 = ring[0]
    for i in range(1, n + 1):
        x1, y1 = ring[i % n]
        # y가 선분 사이에 있고, 교차 여부 계산
        if ((y0 > y) != (y1 > y)):
            x_intersect = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            if x < x_intersect:
                inside = not inside
        x0, y0 = x1, y1

    return inside


def point_in_geom(lon: float, lat: float, geom: dict) -> bool:
    """
    GeoJSON geometry(Polygon/MultiPolygon) 내부 여부
    - 바깥 링만 사용(holes 무시): 행정동 경계는 보통 holes 거의 없어서 실무적으로 OK
    """
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        # coords = [outer_ring, hole1, ...]
        outer = coords[0] if coords else []
        return point_in_polygon(lon, lat, outer)

    if gtype == "MultiPolygon":
        # coords = [ [poly1], [poly2], ... ] where poly = [outer, holes...]
        for poly in coords:
            outer = poly[0] if poly else []
            if point_in_polygon(lon, lat, outer):
                return True
        return False

    return False


def assign_dong_to_points(df_points: pd.DataFrame, admdong_fc: dict, lon_col="lon", lat_col="lat") -> pd.DataFrame:
    """
    df_points의 각 점에 대해 동(adm_nm / dong)을 매칭
    """
    features = admdong_fc.get("features", [])

    def find_dong(row):
        lon = float(row[lon_col])
        lat = float(row[lat_col])

        for f in features:
            props = f.get("properties", {})
            name = str(props.get("adm_nm", "")).strip()
            geom = f.get("geometry", {})
            if point_in_geom(lon, lat, geom):
                short = name.split()[-1] if name else ""
                return short
        return "미분류"

    out = df_points.copy()
    out["dong"] = out.apply(find_dong, axis=1)
    return out



def render_environment_info_below_map():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡 현재 기온", "-3.2°C")
    with col2:
        st.metric("⏰ 현재 시각", datetime.now().strftime("%H:%M"))
    with col3:
        st.metric("🌨 최근 24시간 강설량", "6.5 cm")


def render_map(sites_all, sites_medium, sites_high, df_recent_events):
    boundary_gu = load_boundary_geojson(JONGNO_BOUNDARY_PATH)

    admdong_all = load_boundary_geojson(ADMDONG_PATH)
    admdong_jongno = filter_admdong_by_gu(admdong_all, TARGET_GU)

    layers = [
        layer_gu_outline(boundary_gu),         # ✅ 종로구 구 경계
        layer_dong_outline(admdong_jongno),    # ✅ 종로구 동 경계(얇게)
    ]

    if show_event_hex:
        hex_layer = layer_event_hex(df_recent_events)
        if hex_layer:
            layers.append(hex_layer)

    if show_all_points:
        layers.append(scatter_layer(sites_all, radius=12, color_rgba=[60, 60, 60, 80]))
    if show_medium_points:
        layers.append(scatter_layer(sites_medium, radius=18, color_rgba=[255, 200, 0, 220]))
    if show_high_points:
        layers.append(scatter_layer(sites_high, radius=22, color_rgba=[255, 0, 0, 230]))

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.572,
            longitude=126.98,
            zoom=12.9,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": "SITE: {site_id}\n우선도: {priority}\n이벤트: {event_count}\n카메라 수: {카메라 수}"},
    )
    st.pydeck_chart(deck, use_container_width=True, height=860)


def build_site_hour_table(site_events: pd.DataFrame, site_cams: pd.DataFrame):
    """
    선택 SITE의 CCTV 목록에 '최근 4시간(시간칸 4개 + 추이)' 붙여서 반환
    - site_cams: cameras_df의 site_id 필터 결과
    - site_events: events_with_site에서 site_id 필터 결과
    """
    idx_kst, hour_labels = make_last_4hour_bins_kst()
    pivot = pd.DataFrame(0, index=site_cams["camera_id"], columns=hour_labels)

    if not site_events.empty:
        site_events = site_events.copy()
        site_events["ts_kst"] = to_kst(site_events["ts"])
        site_events["hour_kst"] = site_events["ts_kst"].dt.floor("H")

        cam_hour = (
            site_events.groupby(["camera_id", "hour_kst"])
            .size()
            .reset_index(name="cnt")
        )

        tmp = cam_hour.pivot(index="camera_id", columns="hour_kst", values="cnt").fillna(0).astype(int)
        tmp = tmp.reindex(columns=idx_kst, fill_value=0)
        tmp.columns = hour_labels
        pivot.update(tmp)

    out = site_cams.merge(pivot, left_on="camera_id", right_index=True, how="left")
    out[hour_labels] = out[hour_labels].fillna(0).astype(int)

    def fmt_with_trend(curr: int, prev: int) -> str:
        d = curr - prev
        if d > 0:
            return f"{curr}(🔺{d})"
        elif d < 0:
            return f"{curr}(🔻{abs(d)})"
        else:
            return f"{curr}(▬0)"

    # 원본 보관
    for col in hour_labels:
        out[col + "_n"] = out[col].astype(int)

    for i, col in enumerate(hour_labels):
        if i == 0:
            out[col] = out[col + "_n"].apply(lambda v: f"{v}(▬0)")
        else:
            prev_col = hour_labels[i - 1]
            out[col] = out.apply(lambda r: fmt_with_trend(int(r[col + "_n"]), int(r[prev_col + "_n"])), axis=1)

    return out, hour_labels


# =========================
# Init + Load
# =========================
init_db()

st.title(f"❄️ {TARGET_GU} 고령자 낙상 예방 관제 시스템")
st.caption("지도=좌표 통합(SITE) · 운영/추적=안심주소(CAMERA)")

cameras_df, sites_df = load_cctv_data()

df_events_all = load_events_df(limit=8000)
df_recent = filter_events_by_time(df_events_all)

# =========================
# 동 경계(종로구) 로드 (집계/라벨 공용)
# =========================
ADMDONG_PATH = os.path.join("data", "seoul_admdong.geojson")

admdong_all = load_boundary_geojson(ADMDONG_PATH)
admdong_jongno = filter_admdong_by_gu(admdong_all, TARGET_GU)


# 이벤트는 camera_id(주소 단위)만 인정
camera_ids = set(cameras_df["camera_id"].tolist())
if not df_recent.empty:
    df_recent = df_recent[df_recent["cctv_id"].isin(camera_ids)].copy()

# Camera 우선도/카운트
cam_counts = df_recent.groupby("cctv_id").size().to_dict() if not df_recent.empty else {}
cameras_df["event_count"] = cameras_df["camera_id"].map(lambda x: int(cam_counts.get(x, 0)))
cameras_df["priority"] = cameras_df["event_count"].map(priority_from_count)

# Site 우선도/카운트
events_joined = df_recent.merge(
    cameras_df[["camera_id", "site_id"]],
    left_on="cctv_id",
    right_on="camera_id",
    how="left"
)
site_counts = events_joined.groupby("site_id").size().to_dict() if not events_joined.empty else {}
sites_df["event_count"] = sites_df["site_id"].map(lambda x: int(site_counts.get(x, 0)))
sites_df["priority"] = sites_df["event_count"].map(priority_from_count)
# =========================
# SITE(좌표 통합)에 동(dong) 매핑
# =========================
sites_df = assign_dong_to_points(
    sites_df,
    admdong_jongno,
    lon_col="lon",
    lat_col="lat"
)
# ✅ 종로구 동 폴리곤 준비
admdong_all = load_boundary_geojson(ADMDONG_PATH)
admdong_jongno = filter_admdong_by_gu(admdong_all, TARGET_GU)
dong_polys = build_dong_polygons(admdong_jongno)

# ✅ sites_df에 dong 할당 (미분류도 nearest로 귀속)
sites_df["dong"] = sites_df.apply(lambda r: assign_dong_nearest(r["lat"], r["lon"], dong_polys), axis=1)

sites_high = sites_df[sites_df["priority"] == "High"].copy()
sites_medium = sites_df[sites_df["priority"] == "Medium"].copy()
sites_all = sites_df.copy()

# =========================
# Layout (좌:지도 / 우:요약+그래프+순위+선택 상세)
# =========================
left, right = st.columns([5, 5], gap="large")

with left:
    st.subheader("🗺️ 위험 현황 지도 (High/Medium 중심)")
    render_map(sites_all, sites_medium, sites_high, df_recent)

    # ✅ 환경정보는 지도 아래로
    if show_environment:
        st.divider()
        render_environment_info_below_map()

    st.caption(
        f"누적 기준: {time_window} · "
        f"새로고침: {'OFF' if not auto_refresh else str(refresh_minutes) + '분'} · "
        f"표시: "
        f"{'전체 ' if show_all_points else ''}"
        f"{'Medium ' if show_medium_points else ''}"
        f"{'High ' if show_high_points else ''}"
        f"{'(HEX ON)' if show_event_hex else ''}"
    )

with right:
    st.subheader("🔎 상황 요약")
    high_cnt = int((sites_df["priority"] == "High").sum())
    med_cnt = int((sites_df["priority"] == "Medium").sum())
    total_recent = int(len(df_recent))

    k1, k2, k3,k4 = st.columns(4)
    with k1:
        st.metric("🔴 High (SITE)", f"{high_cnt:,}")
    with k2:
        st.metric("🟠 Medium (SITE)", f"{med_cnt:,}")
    with k3:
        st.metric("최근 이벤트", f"{total_recent:,}")
    with k4:
        st.metric("High 좌표 수", f"{len(sites_high):,}")
    dong_site_stats = (
    sites_df[sites_df["priority"].isin(["High", "Medium"])]
    .groupby(["dong", "priority"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

    if "High" not in dong_site_stats.columns:
        dong_site_stats["High"] = 0
    if "Medium" not in dong_site_stats.columns:
        dong_site_stats["Medium"] = 0

    dong_site_stats["High+Medium"] = dong_site_stats["High"] + dong_site_stats["Medium"]
    dong_site_stats = dong_site_stats.sort_values(
        ["High", "Medium"], ascending=[False, False]
    )

    st.subheader("🏘️ 동별 위험 좌표(SITE) 현황")
    st.dataframe(dong_site_stats, use_container_width=True)
    # ---- 우측 중단: High/Medium 추이 ----
    st.markdown("### 📈 High/Medium 추이 (최근 4시간)")
    # site-hour 기준으로 High/Medium 집계 (단순 but 관제용 충분)
    idx_kst, hour_labels = make_last_4hour_bins_kst()
    trend = pd.DataFrame({"High": [0]*4, "Medium": [0]*4}, index=hour_labels)

    if not df_recent.empty:
        tmp = df_recent.merge(
            cameras_df[["camera_id", "site_id"]],
            left_on="cctv_id",
            right_on="camera_id",
            how="left"
        ).dropna(subset=["site_id"]).copy()
        if not tmp.empty:
            tmp["ts_kst"] = to_kst(tmp["ts"])
            tmp["hour_kst"] = tmp["ts_kst"].dt.floor("H")
            tmp = tmp[tmp["hour_kst"].isin(idx_kst)].copy()

            if not tmp.empty:
                site_hour = tmp.groupby(["site_id", "hour_kst"]).size().reset_index(name="cnt")
                site_hour["priority"] = site_hour["cnt"].map(priority_from_count)

                for i, h in enumerate(idx_kst):
                    one = site_hour[site_hour["hour_kst"] == h]
                    trend.loc[hour_labels[i], "High"] = int(one.loc[one["priority"] == "High", "cnt"].sum())
                    trend.loc[hour_labels[i], "Medium"] = int(one.loc[one["priority"] == "Medium", "cnt"].sum())

    st.line_chart(trend, height=220)

    # ---- 우측 하단: 위험도 순위 + 클릭(선택) 상세 ----
    st.markdown("### ⚠️ 위험도 순위 (조치 우선, High/Medium SITE)")
    risk_sites = sites_df[sites_df["priority"].isin(["High", "Medium"])].copy()

    if risk_sites.empty:
        st.info("현재 시간창 기준으로 High/Medium 위험 지역이 없습니다.")
        selected_site_id = None
    else:
        # High 먼저 나오게 정렬 강제
        order_map = {"High": 0, "Medium": 1, "Low": 2}
        risk_sites["p_rank"] = risk_sites["priority"].map(order_map).fillna(9).astype(int)
        risk_sites = risk_sites.sort_values(["p_rank", "event_count"], ascending=[True, False]).drop(columns=["p_rank"])

        # ✅ "순위표" + "선택" UX
        rank_view = risk_sites[["site_id", "priority", "event_count", "카메라 수"]].copy()
        rank_view = rank_view.rename(columns={"event_count": "최근 이벤트", "카메라 수": "카메라(주소) 수"})
        rank_view = rank_view.reset_index(drop=True)
        rank_view.index = rank_view.index + 1
        st.dataframe(rank_view.head(25), use_container_width=True, height=260)

        # 선택 컨트롤 (table click은 streamlit 기본으로 못 받으니 selectbox로)
        risk_sites["label"] = risk_sites.apply(
            lambda r: f"{r['site_id']} | {r['priority']} | 이벤트 {int(r['event_count'])} | 카메라 {int(r['카메라 수'])}대",
            axis=1
        )
        selected_label = st.selectbox("상세로 볼 위험 지역(SITE) 선택", options=risk_sites["label"].tolist(), index=0)
        selected_site_id = risk_sites.loc[risk_sites["label"] == selected_label, "site_id"].values[0]

    # =========================
    # ✅ 다음 단계: 선택 SITE 상세 (CCTV 목록+시간칸+추이+로그)
    # =========================
    if selected_site_id is not None:
        st.divider()
        st.subheader("📍 선택 위험 지역 상세 (SITE → CCTV 목록/로그)")

        # 해당 SITE의 카메라 목록(주소 단위)
        site_cams = cameras_df[cameras_df["site_id"] == selected_site_id].copy()
        site_cams = site_cams.sort_values(["event_count"], ascending=False).reset_index(drop=True)

        # 이벤트에 site 붙이고 선택 site만
        events_with_site = df_recent.merge(
            cameras_df[["camera_id", "site_id"]],
            left_on="cctv_id",
            right_on="camera_id",
            how="left"
        )
        site_events = events_with_site[events_with_site["site_id"] == selected_site_id].copy()

        # 시간칸 4개 + 추이 붙인 표
        site_table, hour_cols = build_site_hour_table(site_events=site_events, site_cams=site_cams)

        st.markdown("#### 📋 해당 지역 CCTV 목록 (시간대별 4칸 + 추이)")
        show_cols = ["camera_id", "안심 주소", "priority", "event_count"] + hour_cols
        st.dataframe(site_table[show_cols], use_container_width=True, height=260)

        # 로그 볼 CCTV 선택
        cams_with_recent = site_table[site_table["event_count"] > 0]
        cams_for_select = cams_with_recent if not cams_with_recent.empty else site_table

        selected_cam_in_site = st.selectbox(
            "이 지역에서 로그 볼 CCTV 선택",
            options=cams_for_select["camera_id"].tolist(),
            index=0,
            format_func=lambda cid: (
                f"{cams_for_select.loc[cams_for_select['camera_id']==cid,'안심 주소'].values[0]} "
                f"(이벤트 {int(cams_for_select.loc[cams_for_select['camera_id']==cid,'event_count'].values[0])})"
            )
        )

        st.markdown("#### 🧾 선택 CCTV 이벤트 로그 (KST)")
        sel_events_site = df_recent[df_recent["cctv_id"] == selected_cam_in_site].copy()

        if sel_events_site.empty:
            st.info("해당 CCTV에 최근 이벤트가 없습니다.")
        else:
            sel_events_site = sel_events_site.sort_values("ts", ascending=False)
            sel_events_site["ts_kst"] = to_kst(sel_events_site["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                sel_events_site[["ts_kst", "event_type", "confidence", "source_id"]].head(150),
                use_container_width=True,
                height=260
            )

# =========================
# ✅ 테스트 이벤트는 "맨 아래"로 이동
# =========================
st.divider()
st.subheader("🧪 테스트 이벤트 생성 (맨 아래)")
st.caption("모델 연동 전/후, 이벤트 → DB → 지도/집계 흐름만 확인하는 용도입니다.")

selected_camera_id = st.selectbox(
    "CCTV 선택(안심주소 기준)",
    options=cameras_df["camera_id"].tolist(),
    index=0,
    format_func=lambda cid: f"{cameras_df.loc[cameras_df['camera_id']==cid,'안심 주소'].values[0]}",
)
selected_cam_row = cameras_df[cameras_df["camera_id"] == selected_camera_id].iloc[0]

if st.button("선택 CCTV에 낙상 이벤트 저장(테스트)"):
    insert_event(
        lat=float(selected_cam_row["lat"]),
        lon=float(selected_cam_row["lon"]),
        dong=TARGET_GU,
        cctv_id=selected_camera_id,
        event_type="fall",
        confidence=0.9,
        source_id=SOURCE_ID,
    )
    st.success("이벤트 저장 완료")
    st.rerun()

st.divider()
st.info(
    f"""
    본 시스템은 **{TARGET_GU} CCTV 좌표 데이터를 기반으로**,  
    지도는 **좌표 통합(SITE)** 으로 위험 지역을 한눈에 보여주고,  
    운영/추적은 **안심주소(CAMERA) 단위**로 분리하여  
    동일 위치의 여러 CCTV 중 **어느 CCTV에서 이벤트가 발생했는지** 추적 가능하게 설계했습니다.
    """
)
