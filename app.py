import os
from datetime import datetime, timedelta
import json
import hashlib

import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st

# streamlit_autorefresh가 없을 수도 있어서 안전하게 처리
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ✅ db.py 그대로 사용
from db import init_db, insert_event, fetch_events

# =========================
# Page Config
# =========================
st.set_page_config(page_title="고령자 낙상 예방 관제 시스템", layout="wide")

# =========================
# Constants / Paths
# =========================
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
# Sidebar - Navigation
# =========================
st.sidebar.title("🛰️ 관제 메뉴")

MENU = st.sidebar.radio(
    "이동",
    [
        "📊 Overview",
        "🗺️ 실시간 위험 지도",
        "📍 위험 지역 관리(SITE)",
        "🎥 CCTV 관리(안심주소)",
        "📈 이벤트 분석",
        "⚙️ 시스템 설정",
    ],
    index=0
)

st.sidebar.divider()

# =========================
# Sidebar - Global Controls
# =========================
with st.sidebar.expander("🧩 공통 설정", expanded=True):
    time_window = st.radio("누적 기준", ["최근 4시간", "최근 24시간", "최근 72시간"], index=1)

    auto_refresh = st.checkbox("자동 새로고침", value=False)
    refresh_minutes = st.selectbox("새로고침 주기(분)", [1, 5, 10, 30], index=1)

    if st.button("지금 새로고침"):
        st.rerun()

if auto_refresh and st_autorefresh is not None:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# =========================
# Helpers
# =========================
def get_window_hours() -> int:
    return {"최근 4시간": 4, "최근 24시간": 24, "최근 72시간": 72}[time_window]


def to_kst(ts: pd.Series) -> pd.Series:
    """
    DB ts는 UTC 기준으로 저장된다고 가정.
    tz-naive면 UTC로 간주 후 KST로 변환.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("Asia/Seoul")


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
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["ts", "lat", "lon", "cctv_id"]).copy()
    return df


def filter_events_by_time(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(hours=hours)
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


def render_map(
    sites_all: pd.DataFrame,
    sites_medium: pd.DataFrame,
    sites_high: pd.DataFrame,
    df_recent_events: pd.DataFrame,
    show_all_points: bool,
    show_medium_points: bool,
    show_high_points: bool,
    show_event_hex: bool
):
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
    st.pydeck_chart(deck, use_container_width=True, height=780)


def fmt_delta(d: int) -> str:
    if d > 0:
        return f"▲ {d}"
    if d < 0:
        return f"▼ {abs(d)}"
    return "— 0"


def make_last_4hour_bins_kst():
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").floor("H")
    hours_kst = [now_kst - pd.Timedelta(hours=i) for i in range(3, -1, -1)]
    idx_kst = pd.DatetimeIndex(hours_kst)
    labels = [
        f"{h.strftime('%H:%M')}~{(h + pd.Timedelta(hours=1)).strftime('%H:%M')}"
        for h in idx_kst
    ]
    return idx_kst, labels


def fmt_with_trend(curr: int, prev: int) -> str:
    d = curr - prev
    if d > 0:
        return f"{curr}(🔺{d})"
    elif d < 0:
        return f"{curr}(🔻{abs(d)})"
    else:
        return f"{curr}(▬0)"


# =========================
# Init & Load
# =========================
init_db()

st.title(f"❄️ {TARGET_GU} 고령자 낙상 예방 관제 시스템")
st.caption("반복 위험 패턴 기반 사전 예방 관제 · 지도는 좌표 통합(SITE), 운영은 안심주소(CAMERA)")

# 데이터 로드
cameras_df, sites_df = load_cctv_data()
df_events_all = load_events_df(limit=8000)

# 최근(공통 time_window) 필터
HOURS = get_window_hours()
df_recent = filter_events_by_time(df_events_all, HOURS)

# ✅ 이벤트의 cctv_id는 camera_id(주소 단위)만 인정
camera_ids = set(cameras_df["camera_id"].tolist())
if not df_recent.empty:
    df_recent = df_recent[df_recent["cctv_id"].isin(camera_ids)].copy()

# Camera 우선도/카운트 (운영용)
cam_counts = df_recent.groupby("cctv_id").size().to_dict() if not df_recent.empty else {}
cameras_df["event_count"] = cameras_df["camera_id"].map(lambda x: int(cam_counts.get(x, 0)))
cameras_df["priority"] = cameras_df["event_count"].map(priority_from_count)

# Site 우선도/카운트 (지도용)
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
# Common derived metrics (Overview용)
# =========================
def window_compare_counts(df_all: pd.DataFrame, hours: int):
    """
    현재 윈도우(최근 N시간) vs 직전 윈도우(그 이전 N시간) 비교
    ⚠️ ts는 반드시 UTC tz-aware로 맞춘다
    """
    df = df_all.copy()

    # ✅ 핵심: ts를 UTC tz-aware로 통일
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")

    now_utc = pd.Timestamp.now(tz="UTC")
    cur_start = now_utc - pd.Timedelta(hours=hours)
    prev_start = cur_start - pd.Timedelta(hours=hours)

    cur = df[(df["ts"] >= cur_start) & (df["ts"] <= now_utc)].copy()
    prev = df[(df["ts"] >= prev_start) & (df["ts"] < cur_start)].copy()

    # camera_id 필터 동일 적용
    if not cur.empty:
        cur = cur[cur["cctv_id"].isin(camera_ids)].copy()
    if not prev.empty:
        prev = prev[prev["cctv_id"].isin(camera_ids)].copy()

    return cur, prev


cur_w, prev_w = window_compare_counts(df_events_all, HOURS)
cur_events = len(cur_w)
prev_events = len(prev_w)
delta_events = cur_events - prev_events

cur_high_sites = int((sites_df["priority"] == "High").sum())
cur_medium_sites = int((sites_df["priority"] == "Medium").sum())

# 급증(최근 1시간 vs 그 전 1시간) 탐지
last1 = filter_events_by_time(df_events_all, 1)
last2 = filter_events_by_time(df_events_all, 2)
# 1~2시간 구간 = last2 - last1
prev1 = last2[~last2.index.isin(last1.index)].copy()

if not last1.empty:
    last1 = last1[last1["cctv_id"].isin(camera_ids)].copy()
if not prev1.empty:
    prev1 = prev1[prev1["cctv_id"].isin(camera_ids)].copy()

# site 기준으로 합산
last1_site = (
    last1.merge(cameras_df[["camera_id", "site_id"]], left_on="cctv_id", right_on="camera_id", how="left")
    if not last1.empty else pd.DataFrame(columns=["site_id"])
)
prev1_site = (
    prev1.merge(cameras_df[["camera_id", "site_id"]], left_on="cctv_id", right_on="camera_id", how="left")
    if not prev1.empty else pd.DataFrame(columns=["site_id"])
)

last1_counts = last1_site.groupby("site_id").size().to_dict() if not last1_site.empty else {}
prev1_counts = prev1_site.groupby("site_id").size().to_dict() if not prev1_site.empty else {}

surge_sites = []
for sid, c in last1_counts.items():
    p = int(prev1_counts.get(sid, 0))
    if c >= max(3, p * 2) and c - p >= 2:
        surge_sites.append((sid, c, p))

surge_sites = sorted(surge_sites, key=lambda x: (x[1] - x[2], x[1]), reverse=True)


def compute_control_status():
    """
    관제 상태(정상/주의/심각) 간단 룰
    - High가 많거나
    - 최근 1시간 급증이 있거나
    - 전체 이벤트가 급증하면 상향
    """
    score = 0
    if cur_high_sites >= 3:
        score += 2
    elif cur_high_sites >= 1:
        score += 1

    if len(surge_sites) >= 1:
        score += 2

    if delta_events >= 5:
        score += 1
    if cur_events >= 20 and HOURS == 24:
        score += 1

    if score >= 4:
        return "🚨 심각", "error"
    if score >= 2:
        return "⚠️ 주의", "warning"
    return "✅ 정상", "info"


status_text, status_level = compute_control_status()


def status_banner():
    msg = (
        f"**현재 관제 상태: {status_text}**\n\n"
        f"- 누적 기준: **{time_window}**\n"
        f"- 최근 {HOURS}시간 이벤트: **{cur_events:,}건** (직전 구간 대비 {fmt_delta(delta_events)})\n"
        f"- High 위험 지역(SITE): **{cur_high_sites}곳** · Medium: **{cur_medium_sites}곳**\n"
        f"- 최근 1시간 급증 지역: **{len(surge_sites)}곳**"
    )
    if status_level == "error":
        st.error(msg)
    elif status_level == "warning":
        st.warning(msg)
    else:
        st.info(msg)


# =========================
# Pages
# =========================
if MENU == "📊 Overview":
    status_banner()

    # KPI Row (관제형: 변화량 포함)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("최근 이벤트(전체)", f"{cur_events:,}", delta=fmt_delta(delta_events))
    with k2:
        st.metric("High SITE", f"{cur_high_sites:,}")
    with k3:
        st.metric("Medium SITE", f"{cur_medium_sites:,}")
    with k4:
        st.metric("CCTV(안심주소) 수", f"{len(cameras_df):,}")
    with k5:
        avg_conf = float(df_recent["confidence"].dropna().mean()) if not df_recent.empty else 0.0
        st.metric("평균 Confidence", f"{avg_conf:.2f}")

    st.divider()

    # Overview 지도는 관제용으로 단순화: High/Medium만 고정
    st.subheader("🗺️ 위험 현황 지도 (High/Medium 중심)")
    render_map(
        sites_all=sites_all,
        sites_medium=sites_medium,
        sites_high=sites_high,
        df_recent_events=df_recent,
        show_all_points=False,
        show_medium_points=True,
        show_high_points=True,
        show_event_hex=False
    )

    # =========================
    # 지도 연동 위험 지역 패널 (⭐ 관제 UX 핵심 ⭐)
    # =========================
    st.subheader("📍 지도 기준 위험 지역")

    panel_sites = sites_df[sites_df["priority"].isin(["High", "Medium"])].copy()
    panel_sites = panel_sites.sort_values(
        ["priority", "event_count"],
        ascending=[True, False]  # High 먼저, 이벤트 많은 순
    )

    if panel_sites.empty:
        st.caption("현재 지도 기준 위험 지역이 없습니다.")
    else:
        panel_sites["label"] = panel_sites.apply(
            lambda r: f"{r['site_id']} | {r['priority']} | 이벤트 {r['event_count']} | CCTV {r['카메라 수']}대",
            axis=1
        )

        selected_panel_label = st.radio(
            "지도에서 확인한 위험 지역 선택",
            options=panel_sites["label"].tolist(),
            index=0,
            label_visibility="collapsed",
            key="overview_site_panel"
        )

        # 👉 SITE 관리 화면으로 넘길 값
        st.session_state["selected_site_id"] = panel_sites.loc[
            panel_sites["label"] == selected_panel_label, "site_id"
        ].values[0]

        st.caption("선택 시, 📍 위험 지역 관리(SITE) 화면에서 자동으로 상세가 열립니다.")

        st.divider()

        # 실시간 이벤트 타임라인(간단)
        st.subheader("🕒 최근 이벤트 타임라인(시간대별)")
        if df_recent.empty:
            st.info("현재 시간창 기준 이벤트가 없습니다.")
        else:
            df_t = df_recent.copy()
            df_t["ts_kst"] = to_kst(df_t["ts"])
            df_t["hour"] = df_t["ts_kst"].dt.floor("H")
            by_hour = df_t.groupby("hour").size().reset_index(name="count").sort_values("hour")
            by_hour = by_hour.set_index("hour")
            st.line_chart(by_hour["count"])

        # 급증 리스트
        st.subheader("⚡ 최근 1시간 급증 지역")
        if len(surge_sites) == 0:
            st.caption("급증으로 판단된 지역이 없습니다.")
        else:
            surge_df = pd.DataFrame(surge_sites, columns=["site_id", "최근1h", "이전1h"])
            surge_df["증가"] = surge_df["최근1h"] - surge_df["이전1h"]
            st.dataframe(surge_df.head(10), use_container_width=True)

        # 조치 우선 CCTV Top
        st.subheader("⚠️ 조치 우선 CCTV Top 20 (안심주소 단위)")
        list_df = cameras_df[["camera_id", "안심 주소", "event_count", "priority", "site_id"]].copy()
        list_df = list_df.sort_values(["event_count"], ascending=[False]).reset_index(drop=True)
        st.dataframe(list_df.head(20), use_container_width=True)

elif MENU == "🗺️ 실시간 위험 지도":
    st.subheader(f"🗺️ {TARGET_GU} 위험 현황 지도 (실시간 탐색)")

    with st.sidebar.expander("🗺️ 지도 표시 옵션", expanded=True):
        show_all_points = st.checkbox("전체 CCTV 위치 표시", value=False)
        show_high_points = st.checkbox("High 위치 표시", value=True)
        show_medium_points = st.checkbox("Medium 위치 표시", value=True)
        show_event_hex = st.checkbox("이벤트 격자(HEX) 표시", value=False)

    render_map(
        sites_all=sites_all,
        sites_medium=sites_medium,
        sites_high=sites_high,
        df_recent_events=df_recent,
        show_all_points=show_all_points,
        show_medium_points=show_medium_points,
        show_high_points=show_high_points,
        show_event_hex=show_event_hex
    )

    st.caption(
        f"누적 기준: {time_window} · "
        f"자동 새로고침: {'OFF' if not auto_refresh else str(refresh_minutes) + '분'}"
    )

elif MENU == "📍 위험 지역 관리(SITE)":
    st.subheader("📍 위험 지역(SITE) 빠른 탐색 (High/Medium)")
    
    # Overview 지도 패널에서 넘어온 SITE
    preselected_site_id = st.session_state.get("selected_site_id", None)

    risk_sites = sites_df[sites_df["priority"].isin(["High", "Medium"])].copy()
    risk_sites = risk_sites.sort_values(["event_count"], ascending=[False])

    labels = []
    if risk_sites.empty:
        st.info("현재 시간창 기준으로 High/Medium 위험 지역이 없습니다.")
    else:
        risk_sites["label"] = risk_sites.apply(
            lambda r: f"{r['site_id']} | {r['priority']} | 이벤트 {r['event_count']} | 카메라 {r['카메라 수']}대",
            axis=1
        )
        labels = risk_sites["label"].tolist()

        if preselected_site_id:
            match = risk_sites[risk_sites["site_id"] == preselected_site_id]
            default_index = labels.index(match["label"].values[0]) if not match.empty else 0
        else:
            default_index = 0

        selected_site_label = st.selectbox(
            "위험 지역 선택",
            options=labels,
            index=default_index
        )

        selected_site_id = risk_sites.loc[risk_sites["label"] == selected_site_label, "site_id"].values[0]

        st.caption("선택한 SITE에 속한 안심주소 CCTV 목록과, 각 CCTV 이벤트 로그를 빠르게 확인합니다.")

        # 해당 SITE의 CCTV 목록 + 시간대 4칸 + 추이
        site_cams = cameras_df[cameras_df["site_id"] == selected_site_id].copy()
        site_cams = site_cams.sort_values(["event_count"], ascending=False).reset_index(drop=True)

        idx_kst, hour_labels = make_last_4hour_bins_kst()
        pivot = pd.DataFrame(0, index=site_cams["camera_id"], columns=hour_labels)

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

        # 추이 문자열화
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

        # 로그 보기
        st.markdown("### 🧾 CCTV 이벤트 로그 (KST)")
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

        sel_events_site = df_recent[df_recent["cctv_id"] == selected_cam_in_site].copy()
        if sel_events_site.empty:
            st.info("해당 CCTV에 최근 이벤트가 없습니다.")
        else:
            sel_events_site = sel_events_site.sort_values("ts", ascending=False)
            sel_events_site["ts_kst"] = to_kst(sel_events_site["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                sel_events_site[["ts_kst", "event_type", "confidence", "source_id"]].head(200),
                use_container_width=True
            )

elif MENU == "🎥 CCTV 관리(안심주소)":
    st.subheader("🎥 CCTV 관리 (안심주소 단위)")

    # 조치 우선 CCTV 목록
    st.markdown("### ⚠️ 조치 우선 CCTV 목록")
    list_df = cameras_df[["camera_id", "안심 주소", "event_count", "priority", "site_id"]].copy()
    list_df = list_df.sort_values(["event_count"], ascending=[False]).reset_index(drop=True)
    list_df.index = list_df.index + 1

    top_n = st.slider("표에 표시할 상위 N", min_value=20, max_value=500, value=120, step=20)
    st.dataframe(list_df.head(top_n), use_container_width=True)

    st.divider()

    # 테스트 이벤트 생성(관제 느낌: 운영 화면에서!)
    st.markdown("### 🧪 테스트 이벤트 생성(주소 단위 CCTV)")
    selected_camera_id = st.selectbox(
        "CCTV 선택(안심주소 기준)",
        options=cameras_df["camera_id"].tolist(),
        index=0,
        format_func=lambda cid: f"{cid} | {cameras_df.loc[cameras_df['camera_id']==cid,'안심 주소'].values[0]}",
    )
    selected_cam_row = cameras_df[cameras_df["camera_id"] == selected_camera_id].iloc[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        test_conf = st.slider("테스트 confidence", 0.0, 1.0, 0.90, 0.01)
    with c2:
        test_event_type = st.selectbox("event_type", ["fall"], index=0)

    if st.button("선택 CCTV에 이벤트 발생(테스트)"):
        insert_event(
            lat=float(selected_cam_row["lat"]),
            lon=float(selected_cam_row["lon"]),
            dong=TARGET_GU,
            cctv_id=selected_camera_id,   # ✅ camera_id 저장
            event_type=str(test_event_type),
            confidence=float(test_conf),
            source_id=SOURCE_ID,
        )
        st.success("이벤트 저장 완료")
        st.rerun()

    st.divider()

    # 선택 CCTV 상세
    st.markdown("### 📹 선택 CCTV 상세 (웹캠 1프레임 + 로그)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 🎥 웹캠(로컬 CCTV) — 1프레임 캡처")
        cam_on = st.toggle("웹캠 켜기", value=False)
        if cam_on:
            frame, err = webcam_one_frame()
            if err:
                st.error(err)
            else:
                st.image(frame, channels="RGB")
        st.caption("※ 현재는 1프레임 캡처 방식(실시간 스트리밍 X)")

    with right:
        st.markdown("#### 🧾 선택 CCTV 이벤트 로그 (KST)")
        sel_events = df_recent[df_recent["cctv_id"] == selected_camera_id].copy()
        if sel_events.empty:
            st.info("해당 CCTV에 최근 이벤트가 없습니다.")
        else:
            sel_events = sel_events.sort_values("ts", ascending=False)
            sel_events["ts_kst"] = to_kst(sel_events["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                sel_events[["ts_kst", "event_type", "confidence", "source_id"]].head(200),
                use_container_width=True
            )

elif MENU == "📈 이벤트 분석":
    st.subheader("📈 이벤트 분석")

    if df_recent.empty:
        st.info("현재 시간창 기준 이벤트가 없습니다.")
    else:
        df_a = df_recent.copy()
        df_a["ts_kst"] = to_kst(df_a["ts"])
        df_a["hour"] = df_a["ts_kst"].dt.floor("H")

        st.markdown("### ⏱️ 시간대별 이벤트(시간 단위)")
        by_hour = df_a.groupby("hour").size().reset_index(name="count").sort_values("hour").set_index("hour")
        st.line_chart(by_hour["count"])

        st.divider()

        st.markdown("### 🧭 SITE별 이벤트 Top 20")
        a_join = df_a.merge(
            cameras_df[["camera_id", "site_id"]],
            left_on="cctv_id",
            right_on="camera_id",
            how="left"
        )
        by_site = a_join.groupby("site_id").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(by_site.head(20), use_container_width=True)

        st.divider()

        st.markdown("### 🎯 CCTV(안심주소)별 이벤트 Top 30")
        by_cam = df_a.groupby("cctv_id").size().reset_index(name="count").sort_values("count", ascending=False)
        by_cam = by_cam.merge(cameras_df[["camera_id", "안심 주소", "site_id"]], left_on="cctv_id", right_on="camera_id", how="left")
        st.dataframe(by_cam[["cctv_id", "안심 주소", "site_id", "count"]].head(30), use_container_width=True)

elif MENU == "⚙️ 시스템 설정":
    st.subheader("⚙️ 시스템 설정")

    st.markdown("### 🌦️ 환경 정보(데모용)")
    show_environment = st.checkbox("환경 정보 표시", value=True)
    if show_environment:
        render_environment_info()

    st.divider()

    st.markdown("### 🧭 표시 정책(권장값 안내)")
    st.info(
        "- **Overview**: High/Medium 중심(옵션 최소화)\n"
        "- **실시간 지도**: 옵션 제공(HEX/전체 점 등)\n"
        "- **운영(안심주소)**: 조치 우선 목록 + 개별 로그\n"
        "- **SITE 관리**: 위험 지역에서 CCTV로 내려가며 원인 추적"
    )

    st.divider()

    st.markdown("### 🔁 자동 새로고침 상태")
    if auto_refresh:
        st.success(f"자동 새로고침 ON · {refresh_minutes}분 주기")
    else:
        st.warning("자동 새로고침 OFF")

    if st_autorefresh is None:
        st.warning("streamlit_autorefresh가 설치되어 있지 않아 자동 새로고침이 동작하지 않습니다.")

st.divider()
st.info(
    f"""
본 시스템은 **{TARGET_GU} CCTV 좌표 데이터를 기반으로**,  
지도는 **좌표 통합(SITE)** 으로 위험 지역을 한눈에 보여주고,  
운영 화면(목록/상세)은 **안심주소(CAMERA) 단위**로 분리하여  
동일 위치의 여러 CCTV 중 **어느 CCTV에서 이벤트가 발생했는지** 추적 가능하게 설계했습니다.
"""
)
