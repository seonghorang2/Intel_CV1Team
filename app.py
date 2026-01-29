import os
from datetime import datetime, timedelta
import json
import hashlib

import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ✅ db.py 그대로 사용
from db import init_db, insert_event, fetch_events

# =========================
# Page Config
# =========================
st.set_page_config(page_title="고령자 낙상 예방 관제 시스템", layout="wide")

TARGET_GU = "종로구"
CCTV_CSV_PATH = os.path.join("data", "seoul_cctv.csv")
JONGNO_BOUNDARY_PATH = os.path.join("data", "jongno_boundary.geojson")

# ✅ 노트북(웹캠) 고정 좌표 추가
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
# Sidebar - Controls
# =========================
st.sidebar.header("⚙️ 화면 설정")

show_environment = st.sidebar.checkbox("환경 정보 표시", value=True)

st.sidebar.divider()
st.sidebar.subheader("🗺️ CCTV 점 표시 (지도는 좌표 통합)")
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

# =========================
# Helpers
# =========================
def to_kst(ts: pd.Series) -> pd.Series:
    """
    DB ts는 UTC 기준으로 저장된다고 가정.
    tz-naive면 UTC로 간주 후 KST로 변환.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("Asia/Seoul")


def make_last_4hour_bins_kst():
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").floor("H")
    hours_kst = [now_kst - pd.Timedelta(hours=i) for i in range(3, -1, -1)]  # 4칸
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


def layer_gu_outline(geojson_fc: dict):
    # ✅ 히트맵/점 방해 안 하게 얇고 반투명
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_fc,
        stroked=True,
        filled=False,
        get_line_color=[0, 120, 255, 120],
        get_line_width=2,
        # 버전 차이로 에러 나면 아래 2줄 주석 처리
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
    ✅ 2단 구조 반환
    - cameras_df: 주소 단위(중복 좌표 허용) -> 목록/선택/상세용
    - sites_df: 좌표 단위(중복 좌표 통합) -> 지도용
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

    # 노트북(웹캠) 추가: 주소 1개 카메라로 취급
    df = pd.concat([df, pd.DataFrame([LAPTOP_ROW])], ignore_index=True)

    # -------------------------
    # Camera (주소 단위)
    # -------------------------
    cameras = df.copy()
    cameras["lat"] = cameras["위도"].astype(float)
    cameras["lon"] = cameras["경도"].astype(float)
    cameras["camera_id"] = cameras.apply(
        lambda r: make_camera_id(str(r["안심 주소"]), float(r["lat"]), float(r["lon"])),
        axis=1
    )
    cameras = cameras.drop_duplicates(subset=["camera_id"]).reset_index(drop=True)

    # -------------------------
    # Site (좌표 단위 - 지도용 통합)
    # -------------------------
    sites = (
        cameras.groupby(["lat", "lon"], as_index=False)
        .agg(
            {
                "자치구": "first",
                "CCTV 수량": "sum",
                "안심 주소": "count",   # 주소 개수 = 카메라 수
            }
        )
        .rename(columns={"안심 주소": "카메라 수"})
        .copy()
    )
    sites["site_id"] = "SITE_" + sites.index.astype(str).str.zfill(5)
    sites["cctv_id"] = sites["site_id"]  # tooltip 호환용

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


def render_environment_info():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡 현재 기온", "-3.2°C")
    with col2:
        st.metric("⏰ 현재 시각", datetime.now().strftime("%H:%M"))
    with col3:
        st.metric("🌨 최근 24시간 강설량", "6.5 cm")


def webcam_one_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "웹캠을 열 수 없습니다. 다른 앱(Zoom/Teams 등)이 점유 중인지 확인하세요."
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "프레임을 읽지 못했습니다."
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, None


def render_map(sites_all: pd.DataFrame, sites_medium: pd.DataFrame, sites_high: pd.DataFrame, df_recent_events: pd.DataFrame):
    boundary = load_boundary_geojson(JONGNO_BOUNDARY_PATH)

    layers = [layer_gu_outline(boundary)]

    if show_event_hex:
        hex_layer = layer_event_hex(df_recent_events)
        if hex_layer:
            layers.append(hex_layer)

    # ✅ 지도는 좌표 통합(Site)만 표시
    if show_all_points:
        layers.append(scatter_layer(sites_all, radius=12, color_rgba=[60, 60, 60, 90]))
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
    st.pydeck_chart(deck, use_container_width=True, height=850)


def fmt_with_trend(curr: int, prev: int) -> str:
    d = curr - prev
    if d > 0:
        return f"{curr}(🔺{d})"
    elif d < 0:
        return f"{curr}(🔻{abs(d)})"
    else:
        return f"{curr}(▬0)"


# =========================
# Init
# =========================
init_db()

if auto_refresh:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

st.title(f"❄️ {TARGET_GU} 고령자 낙상 예방 관제 시스템")
st.caption("반복 위험 패턴 기반 사전 예방 관제 · 지도는 좌표 통합(SITE), 운영은 안심주소(CAMERA)")

# =========================
# Load data
# =========================
cameras_df, sites_df = load_cctv_data()

df_events_all = load_events_df(limit=8000)
df_recent = filter_events_by_time(df_events_all)

# ✅ 이벤트의 cctv_id는 camera_id(주소 단위)만 인정
camera_ids = set(cameras_df["camera_id"].tolist())
if not df_recent.empty:
    df_recent = df_recent[df_recent["cctv_id"].isin(camera_ids)].copy()

# -------------------------
# Camera 우선도/카운트 (운영용)
# -------------------------
cam_counts = df_recent.groupby("cctv_id").size().to_dict() if not df_recent.empty else {}
cameras_df["event_count"] = cameras_df["camera_id"].map(lambda x: int(cam_counts.get(x, 0)))
cameras_df["priority"] = cameras_df["event_count"].map(priority_from_count)

# -------------------------
# Site 우선도/카운트 (지도용)
# camera -> site 매핑 후 합산
# -------------------------
events_joined = df_recent.merge(
    cameras_df[["camera_id", "site_id"]],
    left_on="cctv_id",
    right_on="camera_id",
    how="left"
)
site_counts = events_joined.groupby("site_id").size().to_dict() if not events_joined.empty else {}
sites_df["event_count"] = sites_df["site_id"].map(lambda x: int(site_counts.get(x, 0)))
sites_df["priority"] = sites_df["event_count"].map(priority_from_count)

sites_high = sites_df[sites_df["priority"] == "High"].copy()
sites_medium = sites_df[sites_df["priority"] == "Medium"].copy()
sites_all = sites_df.copy()

# =========================
# Sidebar - Test event generator (주소 단위)
# =========================
st.sidebar.divider()
st.sidebar.subheader("🧪 테스트 이벤트 생성(주소 단위 CCTV)")

selected_camera_id = st.sidebar.selectbox(
    "CCTV 선택(안심주소 기준)",
    options=cameras_df["camera_id"].tolist(),
    index=0,
    format_func=lambda cid: f"{cid} | {cameras_df.loc[cameras_df['camera_id']==cid,'안심 주소'].values[0]}",
)
selected_cam_row = cameras_df[cameras_df["camera_id"] == selected_camera_id].iloc[0]

if st.sidebar.button("선택 CCTV에 낙상 이벤트 발생(테스트)"):
    insert_event(
        lat=float(selected_cam_row["lat"]),
        lon=float(selected_cam_row["lon"]),
        dong=TARGET_GU,
        cctv_id=selected_camera_id,   # ✅ camera_id 저장
        event_type="fall",
        confidence=0.9,
        source_id=SOURCE_ID,
    )
    st.sidebar.success("이벤트 저장 완료")
    st.rerun()

# =========================
# Main UI
# =========================
if show_environment:
    render_environment_info()
    st.divider()

st.subheader(f"🗺️ {TARGET_GU} 위험 현황 지도 (High/Medium은 좌표 통합 표시)")
render_map(sites_all, sites_medium, sites_high, df_recent)

st.caption(
    f"누적 기준: {time_window} · "
    f"자동 새로고침: {'OFF' if not auto_refresh else str(refresh_minutes) + '분'} · "
    f"표시: "
    f"{'전체 ' if show_all_points else ''}"
    f"{'Medium ' if show_medium_points else ''}"
    f"{'High ' if show_high_points else ''}"
    f"{'(HEX ON)' if show_event_hex else ''}"
)

st.divider()

# KPI
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("CCTV(안심주소) 수", f"{len(cameras_df):,}")
with k2:
    st.metric("지도 표시 좌표(SITE) 수", f"{len(sites_df):,}")
with k3:
    st.metric("최근 이벤트(전체)", f"{len(df_recent):,}")
with k4:
    st.metric("High 좌표 수", f"{len(sites_high):,}")

# =========================
# 조치 우선 CCTV 목록 (주소 단위)
# =========================
st.subheader("⚠️ 조치 우선 CCTV 목록 (안심주소 단위)")
list_df = cameras_df[["camera_id", "안심 주소", "event_count", "priority", "site_id"]].copy()
list_df = list_df.sort_values(["event_count"], ascending=[False]).reset_index(drop=True)
list_df.index = list_df.index + 1

top_n = st.slider("표에 표시할 상위 N", min_value=20, max_value=500, value=120, step=20)
st.dataframe(list_df.head(top_n), use_container_width=True)

st.divider()

# =========================
# 위험 지역(SITE) 기반 빠른 탐색 (High/Medium만)
# =========================
st.subheader("📍 위험 지역(SITE) 빠른 탐색 (High/Medium만)")

risk_sites = sites_df[sites_df["priority"].isin(["High", "Medium"])].copy()
# High가 먼저 보이게: High(=문자) 정렬이 애매하면 event_count 기준으로만 정렬해도 됨
risk_sites = risk_sites.sort_values(["event_count"], ascending=[False])

selected_site_id = None

if risk_sites.empty:
    st.info("현재 시간창 기준으로 High/Medium 위험 지역이 없습니다.")
else:
    risk_sites["label"] = risk_sites.apply(
        lambda r: f"{r['site_id']} | {r['priority']} | 이벤트 {r['event_count']} | 카메라 {r['카메라 수']}대",
        axis=1
    )
    selected_site_label = st.selectbox("위험 지역 선택", options=risk_sites["label"].tolist(), index=0)
    selected_site_id = risk_sites.loc[risk_sites["label"] == selected_site_label, "site_id"].values[0]

    st.caption("선택한 SITE에 속한 안심주소 CCTV 목록과, 각 CCTV 이벤트 로그를 빠르게 확인합니다.")

    # -------------------------
    # 해당 SITE의 CCTV 목록 (주소 단위) + 시간대 4칸 + 추이
    # -------------------------
    site_cams = cameras_df[cameras_df["site_id"] == selected_site_id].copy()
    site_cams = site_cams.sort_values(["event_count"], ascending=False).reset_index(drop=True)

    idx_kst, hour_labels = make_last_4hour_bins_kst()

    # ✅ pivot 기본틀 (이벤트 없어도 절대 오류 안 나게)
    pivot = pd.DataFrame(0, index=site_cams["camera_id"], columns=hour_labels)

    # 이벤트에 site 붙이고 선택 site만
    events_with_site = df_recent.merge(
        cameras_df[["camera_id", "site_id"]],
        left_on="cctv_id",
        right_on="camera_id",
        how="left"
    )
    site_events = events_with_site[events_with_site["site_id"] == selected_site_id].copy()

    if not site_events.empty:
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

    site_cams = site_cams.merge(pivot, left_on="camera_id", right_index=True, how="left")
    site_cams[hour_labels] = site_cams[hour_labels].fillna(0).astype(int)

    # 추이 문자열화 (원본 숫자 유지용 _n)
    for col in hour_labels:
        site_cams[col + "_n"] = site_cams[col].astype(int)

    for i, col in enumerate(hour_labels):
        if i == 0:
            site_cams[col] = site_cams[col + "_n"].apply(lambda v: f"{v}(▬0)")
        else:
            prev_col = hour_labels[i - 1]
            site_cams[col] = site_cams.apply(
                lambda r: fmt_with_trend(int(r[col + "_n"]), int(r[prev_col + "_n"])),
                axis=1
            )

    st.markdown("### 📋 해당 지역 CCTV 목록 (시간대별 + 추이)")
    show_cols = ["camera_id", "안심 주소", "priority", "event_count"] + hour_labels
    st.dataframe(site_cams[show_cols], use_container_width=True)

    # -------------------------
    # 그 중 하나 선택해서 로그 보기
    # -------------------------
    if not site_cams.empty:
        cams_with_recent = site_cams[site_cams["event_count"] > 0]
        cams_for_select = cams_with_recent if not cams_with_recent.empty else site_cams

        selected_cam_in_site = st.selectbox(
            "이 지역에서 로그 볼 CCTV 선택",
            options=cams_for_select["camera_id"].tolist(),
            index=0,
            format_func=lambda cid: (
                f"{cid} | "
                f"{cams_for_select.loc[cams_for_select['camera_id']==cid,'안심 주소'].values[0]} "
                f"(이벤트 {cams_for_select.loc[cams_for_select['camera_id']==cid,'event_count'].values[0]})"
            )
        )

        st.markdown("### 🧾 선택 CCTV 이벤트 로그 (KST)")
        sel_events_site = df_recent[df_recent["cctv_id"] == selected_cam_in_site].copy()

        if sel_events_site.empty:
            st.info("해당 CCTV에 최근 이벤트가 없습니다.")
        else:
            sel_events_site = sel_events_site.sort_values("ts", ascending=False)
            sel_events_site["ts_kst"] = to_kst(sel_events_site["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                sel_events_site[["ts_kst", "event_type", "confidence", "source_id"]].head(100),
                use_container_width=True
            )

st.divider()

# =========================
# 선택 CCTV 상세 (웹캠 1프레임 + 로그)
# =========================
st.subheader("📹 선택 CCTV 상세 (안심주소 단위)")
left, right = st.columns([1, 1])

with left:
    st.markdown("### 🎥 웹캠(로컬 CCTV) — 1프레임 캡처")
    cam_on = st.toggle("웹캠 켜기", value=False)
    if cam_on:
        frame, err = webcam_one_frame()
        if err:
            st.error(err)
        else:
            st.image(frame, channels="RGB")
    st.caption("※ 현재는 1프레임 캡처 방식(실시간 스트리밍 X)")

st.divider()
st.info(
    f"""
    본 시스템은 **{TARGET_GU} CCTV 좌표 데이터를 기반으로**,  
    지도는 **좌표 통합(SITE)** 으로 위험 지역을 한눈에 보여주고,  
    운영 화면(목록/상세)은 **안심주소(CAMERA) 단위**로 분리하여  
    동일 위치의 여러 CCTV 중 **어느 CCTV에서 이벤트가 발생했는지** 추적 가능하게 설계했습니다.
    """
)
