import os
from datetime import datetime, timedelta
import json
import hashlib

import time
import numpy as np
import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import onnxruntime as ort

SEQ_LEN = 30
CONF_TH = 0.65
MIN_FALL_FRAMES = 3   # 연속 fall 프레임 수
COOLDOWN_SEC = 5

# 카메라(노트북 웹캠) 고정 정보
TARGET_GU = "종로구"
LAPTOP_LAT = 37.583266
LAPTOP_LON = 126.966548
SOURCE_ID = "laptop_cam_01"
CAMERA_ADDRESS = "서울특별시 종로구 옥인동 47-264(노트북 웹캠)"

# MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ONNX model
ort_session = ort.InferenceSession(
    "models/fall_lstm.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = ort_session.get_inputs()[0].name

def decode_output(outputs, conf_th):
    out = np.array(outputs[0], dtype=np.float32)  # (1,2)

    p0, p1 = float(out[0, 0]), float(out[0, 1])

    # logits 방어
    s = p0 + p1
    if not (0.98 <= s <= 1.02) or (p0 < 0 or p1 < 0):
        exp = np.exp(out[0] - np.max(out[0]))
        probs = exp / (np.sum(exp) + 1e-9)
        p0, p1 = float(probs[0]), float(probs[1])

    is_fall = (p1 >= conf_th)
    return is_fall, p0, p1


class CCTVVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.seq = []                 # [(135,), ...]
        self.prev_xy = None           # (33,2)
        self.last_event_ts = 0
        self.fall_streak = 0          # ✅ 추가 (없으면 100% 터짐)
        self.frame_idx = 0

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
        )
        self.last_good_img = None
        
        # ✅ 고정 노트북 웹캠의 camera_id (지도/SITE/통계 공통 키)
        self.camera_id = make_camera_id(
            CAMERA_ADDRESS,
            LAPTOP_LAT,
            LAPTOP_LON
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            is_fall = False
            conf_fall = 0.0
            conf_norm = 0.0

            self.frame_idx += 1
            img = frame.to_ndarray(format="bgr24")
            output_img = img.copy()

            # ✅ [추가] 프레임 스킵 (연산 과부하 방지)
            if self.frame_idx % 3 != 0:
                # draw / inference 안 하고 바로 반환
                return av.VideoFrame.from_ndarray(output_img, format="bgr24")

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            if result.pose_landmarks:
                # 👉 draw는 조건 없이 먼저
                mp_draw.draw_landmarks(
                    output_img,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                lm = result.pose_landmarks.landmark

                # 1️⃣ 현재 좌표 (33,2)
                xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)

                # 2️⃣ 속도 (33,2)
                if self.prev_xy is None:
                    velocity = np.zeros_like(xy)
                else:
                    velocity = xy - self.prev_xy

                self.prev_xy = xy.copy()

                # 3️⃣ 신체 중심 (2)
                L_HIP, R_HIP = 23, 24
                center = (xy[L_HIP] + xy[R_HIP]) / 2.0

                # 4️⃣ 신체 높이 (1)
                NOSE = 0
                height = np.array(
                    [xy[NOSE][1] - center[1]],
                    dtype=np.float32
                )

                # 🔥 5️⃣ 최종 feature (135)
                frame_feat = np.concatenate(
                    [
                        xy.flatten(),        # 66
                        velocity.flatten(),  # 66
                        center,              # 2
                        height               # 1
                    ],
                    axis=0
                )  # (135,)

                self.seq.append(frame_feat)
                
                # ✅ 반드시 필요 (슬라이딩 윈도우)
                if len(self.seq) > SEQ_LEN:
                    self.seq.pop(0)
              
                # ✅ 4️⃣ 여기서부터 LSTM inference (빈도 제한)
                if len(self.seq) == SEQ_LEN and self.frame_idx % 2 == 0:
                    x = np.expand_dims(
                        np.array(self.seq, dtype=np.float32),
                        axis=0
                    )  # (1,30,135)

                    outputs = ort_session.run(None, {input_name: x})
                    is_fall, conf_norm, conf_fall = decode_output(outputs, CONF_TH)


                    if is_fall:
                        self.fall_streak += 1
                    else:
                        self.fall_streak = 0

                    now = time.time()
                    if (
                        self.fall_streak >= MIN_FALL_FRAMES
                        and now - self.last_event_ts > COOLDOWN_SEC
                    ):
                        insert_event(
                            lat=LAPTOP_LAT,
                            lon=LAPTOP_LON,
                            dong=TARGET_GU,
                            cctv_id=self.camera_id,
                            event_type="fall",
                            confidence=float(conf_fall),
                            source_id="시연용 웹캠",
                        )
                        self.last_event_ts = now
                        self.fall_streak = 0

            # 상태 텍스트 결정 (⭐ 수정안 3 핵심)
            if self.fall_streak >= MIN_FALL_FRAMES:
                status_text = "FALL DETECTED"
                status_color = (0, 0, 255)
            elif self.fall_streak > 0:
                status_text = f"ANALYZING ({self.fall_streak}/{MIN_FALL_FRAMES})"
                status_color = (0, 255, 255)
            else:
                status_text = "NORMAL"
                status_color = (0, 255, 0)


            cv2.putText(
                output_img,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                status_color,
                3
            )


            self.last_good_img = output_img
            return av.VideoFrame.from_ndarray(output_img, format="bgr24")


        except Exception as e:
            print("WebRTC recv error:", e)

            if self.last_good_img is not None:
                return av.VideoFrame.from_ndarray(
                    self.last_good_img,
                    format="bgr24"
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")





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
st.set_page_config(page_title="빙판길 위험 관제 시스템(AIS)", layout="wide")

# =========================
# Constants / Paths
# =========================
TARGET_GU = "종로구"
CCTV_CSV_PATH = os.path.join("data", "seoul_cctv.csv")
JONGNO_BOUNDARY_PATH = os.path.join("data", "jongno_boundary.geojson")
ADMDONG_PATH = os.path.join("data", "seoul_admdong.geojson")

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
        "🗺️ 상세현황",
        "⚙️ 시연용 메뉴",
    ],
    index=0
)

st.sidebar.divider()

# =========================
# Sidebar - Global Controls
# =========================
with st.sidebar.expander("🧩 공통 설정", expanded=True):
    if st.button("지금 새로고침"):
        st.rerun()

    time_window = st.radio(
        "누적 기준",
        [
            "최근 4시간",
            "최근 16시간",
            "최근 24시간",
            "최근 48시간",
        ],
        index=0
    )


    auto_refresh = st.checkbox("자동 새로고침", value=False)
    refresh_minutes = st.selectbox("새로고침 주기(분)", [1, 5, 10, 30], index=1)

if auto_refresh and st_autorefresh is not None:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# =========================
# Helpers
# =========================

@st.cache_data
def build_dong_polygons(admdong_jongno_fc: dict):
    """
    return: list of (dong_name, shapely_polygon)
    """
    from shapely.geometry import shape

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
    1) 폴리곤 내부면 해당 동
    2) 아니면 가장 가까운 동으로 귀속
    """
    from shapely.geometry import Point

    p = Point(lon, lat)

    # 1️⃣ contains 우선
    for dong, poly in dong_polys:
        if poly.contains(p):
            return dong

    # 2️⃣ 가장 가까운 동
    best_dong = "미분류"
    best_dist = float("inf")
    for dong, poly in dong_polys:
        d = poly.distance(p)
        if d < best_dist:
            best_dist = d
            best_dong = dong

    return best_dong


def filter_admdong_by_gu(geojson_fc: dict, target_gu: str) -> dict:
    kept = []
    for f in geojson_fc.get("features", []):
        props = f.get("properties", {})
        adm_nm = str(props.get("adm_nm", "")).strip()  # "서울특별시 종로구 사직동"
        if target_gu in adm_nm:
            kept.append(f)

    return {
        "type": "FeatureCollection",
        "features": kept,
    }

def get_window_hours() -> int:
    return {
        "최근 4시간": 4,
        "최근 16시간": 16,
        "최근 24시간": 24,
        "최근 48시간": 48,
    }[time_window]

def colored_kpi(label, value, color, sub=None):
    st.markdown(
        f"""
        <div style="
            border-radius:12px;
            padding:14px;
            background-color:{color};
            color:white;
            margin-bottom:8px;
        ">
            <div style="font-size:13px; opacity:0.9;">{label}</div>
            <div style="font-size:26px; font-weight:700;">{value}</div>
            {f'<div style="font-size:12px; opacity:0.85;">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

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

def scroll_to_top():
    st.markdown(
        """
        <script>
            window.scrollTo({ top: 0, behavior: "instant" });
        </script>
        """,
        unsafe_allow_html=True
    )

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

def layer_dong_outline(geojson_fc: dict):
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_fc,
        stroked=True,
        filled=False,                 # ⭐ 외곽선만
        get_line_color=[80, 80, 80, 100],  # 연회색
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=2,
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
    focus_site: dict | None = None,
):

    boundary = load_boundary_geojson(JONGNO_BOUNDARY_PATH)

    # 🔥 동 경계 로드 + 종로구만 필터
    admdong_all = load_boundary_geojson(ADMDONG_PATH)
    admdong_jongno = filter_admdong_by_gu(admdong_all, TARGET_GU)

    layers = [
        layer_gu_outline(boundary),            # 구 외곽선
        layer_dong_outline(admdong_jongno),    # ⭐ 동 외곽선
    ]


    if show_all_points:
        layers.append(scatter_layer(sites_all, radius=12, color_rgba=[60, 60, 60, 90]))
    if show_medium_points:
        layers.append(scatter_layer(sites_medium, radius=18, color_rgba=[255, 200, 0, 220]))
    if show_high_points:
        layers.append(scatter_layer(sites_high, radius=22, color_rgba=[255, 0, 0, 230]))

    # =========================
    # ✅ SITE 선택 시 지도 중심 이동 + 줌
    # =========================
    if focus_site is not None:
        view_state = pdk.ViewState(
            latitude=focus_site["lat"],
            longitude=focus_site["lon"],
            zoom=15.8,
            pitch=0,
        )
    else:
        view_state = pdk.ViewState(
            latitude=37.572,
            longitude=126.98,
            zoom=12.9,
            pitch=0,
        )

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "SITE: {site_id}\n우선도: {priority}\n이벤트: {event_count}\n카메라 수: {카메라 수}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


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

def build_cctv_timeseries_table(
    site_id: str,
    cameras_df: pd.DataFrame,
    df_recent: pd.DataFrame,
):
    site_cams = cameras_df[cameras_df["site_id"] == site_id].copy()
    base_camera_ids = site_cams["camera_id"].tolist()
    site_cams = site_cams.sort_values("event_count", ascending=False).reset_index(drop=True)

    idx_kst, hour_labels = make_last_4hour_bins_kst()
    pivot = pd.DataFrame(0, index=site_cams["camera_id"], columns=hour_labels)

    events_with_site = df_recent.merge(
        cameras_df[["camera_id", "site_id"]],
        left_on="cctv_id",
        right_on="camera_id",
        how="left"
    )
    site_events = events_with_site[events_with_site["site_id"] == site_id].copy()

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

    # ✅ CCTV 개수만큼만 행 유지 (정답 코드)
    site_cams = (
        site_cams
        .set_index("camera_id")
        .reindex(base_camera_ids)
        .reset_index()
    )


    return site_cams, hour_labels


def fmt_with_trend(curr: int, prev: int) -> str:
    d = curr - prev
    if d > 0:
        return f"{curr}(🔺{d})"
    elif d < 0:
        return f"{curr}(🔻{abs(d)})"
    else:
        return f"{curr}(▬0)"


def render_site_quick_panel(
    sites_df_in: pd.DataFrame,
    cameras_df_in: pd.DataFrame,
    df_recent_in: pd.DataFrame,
    preselected_site_id: str | None = None,
    panel_title: str = "📍 선택 SITE 상세"
):
    """
    ✅ 최소 수정으로 '지도 페이지'에서 SITE → CCTV → 로그까지 한 화면에서 보이게 하는 패널
    - 지도 클릭을 직접 받기보다, 지도 옆/아래 패널에서 SITE를 선택하는 관제 UX
    """
    st.markdown(f"### {panel_title}")

    risk_sites = sites_df_in[sites_df_in["priority"].isin(["High", "Medium"])].copy()
    risk_sites = risk_sites.sort_values(["priority", "event_count"], ascending=[True, False])

    if risk_sites.empty:
        st.info("현재 시간창 기준으로 High/Medium 위험 지역이 없습니다.")
        return

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
        "SITE 선택",
        options=labels,
        index=default_index,
        key="map_panel_site_select"
    )
    selected_site_id = risk_sites.loc[risk_sites["label"] == selected_site_label, "site_id"].values[0]

    # 세션에 저장(다른 메뉴에서도 이어보기)
    st.session_state["selected_site_id"] = selected_site_id

    st.caption("선택한 SITE에 속한 CCTV 목록과, 각 CCTV 이벤트 로그를 바로 확인합니다.")

# =========================
# Init & Load
# =========================
init_db()

st.title("❄️ 빙판길 위험 관제 시스템(AIS)")
st.caption("종로구")

# 데이터 로드
cameras_df, sites_df = load_cctv_data()
df_events_all = load_events_df(limit=8000)

# =========================
# 🔥 행정동 폴리곤 준비 (dong 매핑용)
# =========================
admdong_all = load_boundary_geojson(ADMDONG_PATH)
admdong_jongno = filter_admdong_by_gu(admdong_all, TARGET_GU)
dong_polys = build_dong_polygons(admdong_jongno)

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

# =========================
# SITE(좌표 통합)에 동(dong) 매핑  ← ★ 추가 위치
# =========================
sites_df["dong"] = sites_df.apply(
    lambda r: assign_dong_nearest(r["lat"], r["lon"], dong_polys),
    axis=1
)


sites_high = sites_df[sites_df["priority"] == "High"].copy()
sites_medium = sites_df[sites_df["priority"] == "Medium"].copy()
sites_all = sites_df.copy()
# =========================
# 🏘️ 동별 위험 좌표(SITE) 현황 (행정동 기준, SITE 단위)
# =========================
dong_site_stats = (
    sites_df[sites_df["priority"].isin(["High", "Medium"])]
    .groupby(["dong", "priority"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)


# 컬럼 방어
if "High" not in dong_site_stats.columns:
    dong_site_stats["High"] = 0
if "Medium" not in dong_site_stats.columns:
    dong_site_stats["Medium"] = 0

dong_site_stats["High+Medium"] = (
    dong_site_stats["High"] + dong_site_stats["Medium"]
)

dong_site_stats = dong_site_stats.sort_values(
    ["High", "Medium"],
    ascending=[False, False]
)




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
    msg = f"**현재 관제 상태: {status_text}**"
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
    # =========================
    # Overview 최상단: 지도 + 정보 2분할
    # =========================
    left, right = st.columns([2.3, 1.2], gap="large")

    with left:
        render_map(
            sites_all=sites_all,
            sites_medium=sites_medium,
            sites_high=sites_high,
            df_recent_events=df_recent,
            show_all_points=False,
            show_medium_points=True,
            show_high_points=True,
        )
        st.divider()

        # =========================
        # 📍 현재 위험 지역(Top10)
        # =========================
        st.subheader("📍 위험 지역(Top10)")

        panel_sites = sites_df[sites_df["priority"].isin(["High", "Medium"])].copy()
        panel_sites = panel_sites.sort_values(
            ["priority", "event_count"],
            ascending=[True, False]
        ).head(10)

        if panel_sites.empty:
            st.caption("현재 지도 기준 위험 지역이 없습니다.")
        else:
            panel_sites["label"] = panel_sites.apply(
                lambda r: f"{r['site_id']} | {r['priority']} | 이벤트 {r['event_count']} | CCTV {r['카메라 수']}대",
                axis=1
            )

            selected_panel_label = st.radio(
                "위험 지역 선택",
                options=panel_sites["label"].tolist(),
                index=0,
                label_visibility="collapsed",
                key="overview_site_panel"
            )

            st.session_state["selected_site_id"] = panel_sites.loc[
                panel_sites["label"] == selected_panel_label, "site_id"
            ].values[0]


    with right:
        # =========================
        # 환경 정보 (오른쪽 패널)
        # =========================
        show_environment = st.checkbox("환경 정보 표시", value=False)

        if show_environment:
            render_environment_info()
            st.divider()

        # =========================
        # 관제 상태
        # =========================
        status_banner()
        st.divider()

        # =========================
        # KPI 요약
        # =========================
        k1, k2 = st.columns(2)

        # 색 결정 로직
        event_color = "#d9534f" if delta_events > 0 else "#5cb85c"
        high_color = "#d9534f" if cur_high_sites > 0 else "#5cb85c"
        medium_color = "#f0ad4e" if cur_medium_sites > 0 else "#5cb85c"
        surge_color = "#d9534f" if len(surge_sites) > 0 else "#5cb85c"

        avg_conf = float(df_recent["confidence"].dropna().mean()) if not df_recent.empty else 0.0
        conf_color = "#5cb85c" if avg_conf >= 0.85 else "#f0ad4e"

        with k1:
            colored_kpi(
                f"{time_window} 이벤트",
                f"{cur_events:,}",
                event_color,
                sub=f"직전 대비 {fmt_delta(delta_events)}"
            )
            colored_kpi(
                "High SITE",
                f"{cur_high_sites}곳",
                high_color
            )
            colored_kpi(
                "Medium SITE",
                f"{cur_medium_sites}곳",
                medium_color
            )

        with k2:
            colored_kpi(
                "최근 1시간 급등 지역",
                f"{len(surge_sites)}곳",
                surge_color
            )
            colored_kpi(
                "평균 Confidence",
                f"{avg_conf:.2f}",
                conf_color
            )

        st.divider()

        # =========================
        # 🏘️ 동별 위험 좌표(SITE) 현황
        # =========================
        st.subheader("🏘️ 동별 현황")
        st.dataframe(
            dong_site_stats,
            use_container_width=True,
            height=260
        )

        st.caption("🗺️ 상세현황 확인 필요")

    # =========================
    # 📊 Overview 하단 (전체 폭)
    # =========================
    st.divider()

    # -------------------------
    # 🕒 최근 이벤트 타임라인
    # -------------------------
    st.subheader("🕒 낙상 발생 추이 (시간당)")

    if df_recent.empty:
        st.info("현재 시간창 기준 이벤트가 없습니다.")
    else:
        df_t = df_recent.copy()
        df_t["ts_kst"] = to_kst(df_t["ts"])
        df_t["hour"] = df_t["ts_kst"].dt.tz_localize(None).dt.floor("H")

        by_hour = (
            df_t.groupby("hour")
            .size()
            .reset_index(name="count")
            .sort_values("hour")
        )


        by_hour["time_range"] = by_hour["hour"].apply(
            lambda h: f"{h.strftime('%H:%M')}-{(h + pd.Timedelta(hours=1)).strftime('%H:%M')}"
        )

        by_hour = by_hour.set_index("time_range")[["count"]]

        st.line_chart(by_hour, use_container_width=True)

    st.divider()

    # -------------------------
    # ⚡ 최근 1시간 급증 지역
    # -------------------------
    st.subheader("⚡ 최근 1시간 급증 지역")

    if len(surge_sites) == 0:
        st.caption("급증으로 판단된 지역이 없습니다.")
    else:
        surge_df = pd.DataFrame(
            surge_sites,
            columns=["site_id", "최근 1시간", "이전 1시간"]
        )
        surge_df["증가량"] = surge_df["최근 1시간"] - surge_df["이전 1시간"]

        st.dataframe(
            surge_df.head(10),
            use_container_width=True,
            height=260
        )


elif MENU == "🗺️ 상세현황":
    st.subheader(f"🗺️ {TARGET_GU} 위험 현황 지도")


    with st.sidebar.expander("🗺️ 지도 표시 옵션", expanded=True):
        show_all_points = st.checkbox("전체 CCTV 위치 표시", value=False)
        show_high_points = st.checkbox("High 위치 표시", value=True)
        show_medium_points = st.checkbox("Medium 위치 표시", value=True)

    # ✅ 최소 수정: 지도 + 우측 패널(상세)로 구조 변경
    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        # =========================
        # ✅ 선택된 SITE → 지도 포커스 좌표
        # =========================
        focus_site = None
        selected_site_id = st.session_state.get("selected_site_id", None)

        if selected_site_id:
            row = sites_df[sites_df["site_id"] == selected_site_id]
            if not row.empty:
                focus_site = {
                    "lat": float(row.iloc[0]["lat"]),
                    "lon": float(row.iloc[0]["lon"]),
                }

        render_map(
            sites_all=sites_all,
            sites_medium=sites_medium,
            sites_high=sites_high,
            df_recent_events=df_recent,
            show_all_points=show_all_points,
            show_medium_points=show_medium_points,
            show_high_points=show_high_points,
            focus_site=focus_site,   # ✅ [추가]
        )

        st.caption(
            f"누적 기준: {time_window} · "
            f"자동 새로고침: {'OFF' if not auto_refresh else str(refresh_minutes) + '분'}"
        )
        
    with right:
        preselected_site_id = st.session_state.get("selected_site_id", None)
        render_site_quick_panel(
            sites_df_in=sites_df,
            cameras_df_in=cameras_df,
            df_recent_in=df_recent,
            preselected_site_id=preselected_site_id,
            panel_title="📍 구역 상세 정보 확인"
        )
        
    st.divider()
    st.subheader("📋 구역별 CCTV 상세 현황")

    selected_site_id = st.session_state.get("selected_site_id", None)

    if not selected_site_id:
        st.info("지도 또는 우측 패널에서 SITE를 선택하면 CCTV 상세가 표시됩니다.")
    else:
        # 1️⃣ CCTV 목록 (시간대별 + 추이)
        st.markdown("### 🎥 CCTV 목록 및 시간대별 추이")

        site_cams_ts, hour_labels = build_cctv_timeseries_table(
            selected_site_id,
            cameras_df,
            df_recent
        )

        show_cols = ["camera_id", "안심 주소", "priority", "event_count"] + hour_labels
        st.dataframe(
            site_cams_ts[show_cols],
            use_container_width=True
        )


        st.divider()

        # =========================
        # CCTV 이벤트 로그 + 시연용 CCTV (2분할)
        # =========================
        st.markdown("### 🧾 낙상발생 로그 · CCTV 확인")

        log_col, cam_col = st.columns([1.2, 1.0], gap="large")

        # -------------------------
        # 왼쪽: CCTV 이벤트 로그
        # -------------------------
        with log_col:
            cams_with_recent = site_cams_ts[site_cams_ts["event_count"] > 0]
            cams_for_select = cams_with_recent if not cams_with_recent.empty else site_cams_ts

            selected_cam = st.selectbox(
                "로그 확인할 CCTV",
                options=cams_for_select["camera_id"].tolist(),
                index=0,
                key="map_bottom_cam_select",
                format_func=lambda cid: (
                    f"{cid} | "
                    f"{cams_for_select.loc[cams_for_select['camera_id']==cid,'안심 주소'].values[0]}"
                )
            )

            log_df = df_recent[df_recent["cctv_id"] == selected_cam].copy()

            if log_df.empty:
                st.info("선택한 CCTV에 최근 이벤트가 없습니다.")
            else:
                log_df = log_df.sort_values("ts", ascending=False)
                log_df["ts_kst"] = to_kst(log_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

                st.dataframe(
                    log_df[["ts_kst"]],
                    use_container_width=True,
                    height=320
                )

        # -------------------------
        # 오른쪽: CCTV 화면
        # -------------------------
        with cam_col:
            st.markdown("#### 🎥 CCTV 화면")

            webrtc_streamer(
                key="demo-cctv",
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                async_processing=False,
            )

            st.caption("※ 시연용 웹캠 화면 (실제 CCTV 연동 아님)")

elif MENU == "⚙️ 시연용 메뉴":
    st.subheader("⚙️ 시연용 메뉴")

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
        
    def generate_demo_events_scenario(
        cameras_df: pd.DataFrame,
        high_sites_count: int = 6,
        super_site_events: int = 18,
    ):
        import random

        site_groups = cameras_df.groupby("site_id")
        site_ids = list(site_groups.groups.keys())
        random.shuffle(site_ids)

        super_site_id = site_ids[0]
        high_site_ids = site_ids[1:high_sites_count + 1]

        now = datetime.utcnow()
        inserted = 0

        def random_morning_time():
            base = now - timedelta(days=random.choice([0, 1]))
            hour = random.randint(7, 10)
            minute = random.randint(0, 59)
            return base.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # 🔥 초위험 SITE
        cams = site_groups.get_group(super_site_id)
        for _, cam in cams.iterrows():
            for _ in range(super_site_events // len(cams)):
                insert_event(
                    lat=float(cam["lat"]),
                    lon=float(cam["lon"]),
                    dong=TARGET_GU,
                    cctv_id=cam["camera_id"],
                    event_type="fall",
                    confidence=round(random.uniform(0.85, 0.98), 2),
                    source_id="demo_super_site",
                )
                inserted += 1

        # ⚠️ High SITE들
        for site_id in high_site_ids:
            cams = site_groups.get_group(site_id)
            for _ in range(random.randint(3, 5)):
                cam = cams.sample(1).iloc[0]
                insert_event(
                    lat=float(cam["lat"]),
                    lon=float(cam["lon"]),
                    dong=TARGET_GU,
                    cctv_id=cam["camera_id"],
                    event_type="fall",
                    confidence=round(random.uniform(0.75, 0.95), 2),
                    source_id="demo_high_site",
                )
                inserted += 1

        return super_site_id, inserted
    st.divider()
    st.markdown("### 🎬 시연 연출용(강조 버전)")

    if st.button("🔥 High SITE 집중 · 초위험 지역 연출"):
        with st.spinner("시연 연출 데이터 생성 중..."):
            super_site_id, n = generate_demo_events_scenario(
                cameras_df=cameras_df,
                high_sites_count=6,
                super_site_events=18,
            )

        st.success(
            f"""
    시연 연출 완료

    - 🔥 초위험 SITE: {super_site_id}
    - ⚠️ High SITE 집중 생성
    - 총 이벤트 수: {n}건
    """
        )
        st.rerun()        
        
       

st.divider()
st.info(
    f"""
본 시스템은 **{TARGET_GU} CCTV 좌표 데이터를 기반으로** **위험 지역**을 한눈에 보여주고,  
**어느 CCTV에서 이벤트가 발생했는지** 추적 가능하게 설계했습니다.
"""
)
