import os
from datetime import datetime, timedelta

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
st.sidebar.subheader("🗺️ CCTV 점 표시")
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
def get_window_hours() -> int:
    return {"최근 4시간": 4, "최근 24시간": 24, "최근 72시간": 72}[time_window]


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="euc-kr")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


@st.cache_data
def load_cctv_master() -> pd.DataFrame:
    """
    CSV 컬럼: 자치구, 안심 주소, 위도, 경도, CCTV 수량, 수정 일시
    - 강남구 필터
    - 위경도 정리
    - 노트북 좌표 추가
    - 동일 좌표 합치기(성능/겹침 완화)
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
    df = df.dropna(subset=["위도", "경도"]).copy()

    # 노트북 좌표 추가
    df = pd.concat([df, pd.DataFrame([LAPTOP_ROW])], ignore_index=True)

    # 동일 좌표 합치기
    agg = (
        df.groupby(["위도", "경도"], as_index=False)
        .agg(
            {
                "CCTV 수량": "sum",
                "안심 주소": "first",
                "자치구": "first",
                "수정 일시": "max",
            }
        )
        .copy()
    )

    # 좌표 기반 id 생성 (고유)
    agg["cctv_id"] = "CCTV_" + agg.index.astype(str).str.zfill(5)

    # pydeck용
    agg["lat"] = agg["위도"].astype(float)
    agg["lon"] = agg["경도"].astype(float)

    return agg


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
    cutoff = datetime.utcnow() - timedelta(hours=get_window_hours())  # DB는 UTC 저장 기준
    return df[df["ts"] >= pd.Timestamp(cutoff)].copy()


def priority_from_count(n: int) -> str:
    # 단순/직관 (원하면 기준 바꿔도 됨)
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
    """
    Heatmap은 퍼져서 침범 느낌이 생김 → Hexagon(격자 집계) 사용
    """
    if df_events.empty:
        return None
    return pdk.Layer(
        "HexagonLayer",
        data=df_events,
        get_position="[lon, lat]",
        radius=35,          # 더 촘촘하게 보이게 하려면 25~35 추천
        elevation_scale=0,  # 2D
        extruded=False,
        pickable=True,
    )


def render_environment_info():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡 현재 기온", "-3.2°C")  # TODO: 실제 연동 가능
    with col2:
        st.metric("⏰ 현재 시각", datetime.now().strftime("%H:%M"))
    with col3:
        st.metric("🌨 최근 24시간 강설량", "6.5 cm")  # TODO: 실제 연동 가능


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


def render_map(cctv_all: pd.DataFrame, cctv_medium: pd.DataFrame, cctv_high: pd.DataFrame, df_recent_events: pd.DataFrame):
    layers = []

    if show_event_hex:
        hex_layer = layer_event_hex(df_recent_events)
        if hex_layer:
            layers.append(hex_layer)

    # ✅ 점 레이어 토글
    if show_all_points:
        layers.append(scatter_layer(cctv_all, radius=12, color_rgba=[60, 60, 60, 90]))         # 전체(연회색)
    if show_medium_points:
        layers.append(scatter_layer(cctv_medium, radius=18, color_rgba=[255, 200, 0, 220]))    # Medium(노랑)
    if show_high_points:
        layers.append(scatter_layer(cctv_high, radius=22, color_rgba=[255, 0, 0, 230]))        # High(빨강)

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=LAPTOP_LAT, longitude=LAPTOP_LON, zoom=13, pitch=0),
        layers=layers,
        tooltip={"text": "ID: {cctv_id}\n우선도: {priority}\n이벤트: {event_count}\n주소: {안심 주소}\n수량: {CCTV 수량}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


# =========================
# Init
# =========================
init_db()

# ✅ 자동 새로고침(원할 때만)
if auto_refresh:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

st.title(f"❄️ {TARGET_GU} 고령자 낙상 예방 관제 시스템")
st.caption("반복 위험 패턴 기반 사전 예방 관제 · CCTV 좌표 기반 파일럿")

# =========================
# Load data
# =========================
cctv_master = load_cctv_master()

df_events_all = load_events_df(limit=8000)
df_recent = filter_events_by_time(df_events_all)

# ✅ CSV에 있는 CCTV만 이벤트 인정
master_ids = set(cctv_master["cctv_id"].tolist())
if not df_recent.empty:
    df_recent = df_recent[df_recent["cctv_id"].isin(master_ids)].copy()

# =========================
# Compute priority per CCTV
# =========================
counts = df_recent.groupby("cctv_id").size().to_dict() if not df_recent.empty else {}
cctv_master["event_count"] = cctv_master["cctv_id"].map(lambda x: int(counts.get(x, 0)))
cctv_master["priority"] = cctv_master["event_count"].map(priority_from_count)

cctv_high = cctv_master[cctv_master["priority"] == "High"].copy()
cctv_medium = cctv_master[cctv_master["priority"] == "Medium"].copy()
cctv_all = cctv_master.copy()

# =========================
# Sidebar - Test event generator
# =========================
st.sidebar.divider()
st.sidebar.subheader("🧪 테스트 이벤트 생성(모델 대체)")

selected_id = st.sidebar.selectbox(
    "CCTV 선택(검색 가능)",
    options=cctv_master["cctv_id"].tolist(),
    index=0,
    format_func=lambda cid: f"{cid} | {cctv_master.loc[cctv_master['cctv_id']==cid,'안심 주소'].values[0]}",
)
selected_row = cctv_master[cctv_master["cctv_id"] == selected_id].iloc[0]

if st.sidebar.button("선택 CCTV에 낙상 이벤트 발생(테스트)"):
    insert_event(
        lat=float(selected_row["lat"]),
        lon=float(selected_row["lon"]),
        dong=TARGET_GU,  # CSV에 행정동이 없어서 구 단위로 저장
        cctv_id=selected_id,
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

st.subheader(f"🗺️ {TARGET_GU} CCTV 위험 현황 지도")
render_map(cctv_all, cctv_medium, cctv_high, df_recent)

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
    st.metric("CCTV 좌표 수(중복합침)", f"{len(cctv_master):,}")
with k2:
    st.metric("최근 이벤트(전체)", f"{len(df_recent):,}")
with k3:
    st.metric("High", f"{len(cctv_high):,}")
with k4:
    st.metric("Medium", f"{len(cctv_medium):,}")

st.subheader("⚠️ 조치 우선 CCTV 목록(최근 이벤트 수 기반)")

list_df = cctv_master[["cctv_id", "안심 주소", "CCTV 수량", "event_count", "priority", "lat", "lon"]].copy()
list_df = list_df.sort_values(["event_count", "CCTV 수량"], ascending=[False, False]).reset_index(drop=True)

top_n = st.slider("표에 표시할 상위 N", min_value=20, max_value=300, value=80, step=20)
st.dataframe(list_df.head(top_n), use_container_width=True)

st.divider()

st.subheader("📹 선택 CCTV 상세")
left, right = st.columns([1, 1])

with left:
    st.markdown("### 🎥 웹캠(로컬 CCTV)")
    cam_on = st.toggle("웹캠 켜기", value=False)
    if cam_on:
        frame, err = webcam_one_frame()
        if err:
            st.error(err)
        else:
            st.image(frame, channels="RGB")
    st.caption("※ 실제 모델 연결 전: 사이드바 버튼으로 이벤트를 생성해 지도/집계 흐름을 검증합니다.")

with right:
    st.markdown("### 🧾 선택 CCTV 이벤트 로그")
    sel_events = df_recent[df_recent["cctv_id"] == selected_id].copy()
    if sel_events.empty:
        st.info("최근 시간창 기준으로 이벤트가 없습니다. 사이드바에서 테스트 이벤트를 눌러보세요.")
    else:
        sel_events = sel_events.sort_values("ts", ascending=False)
        st.dataframe(sel_events[["ts", "event_type", "confidence", "source_id"]].head(50), use_container_width=True)

st.divider()
st.info(
    """
    본 시스템은 **강남구 CCTV 좌표 데이터를 기반으로**,  
    반복 위험 이벤트(낙상/빙판)를 집계하여 **조치 우선순위를 판단**하기 위한  
    고령자 낙상 사고 사전 예방 관제 솔루션(MVP)입니다.
    """
)
