import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle, Wedge
from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KBO 선수 기록실", page_icon="⚾", layout="wide")


@st.cache_resource
def setup_font():
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return True


setup_font()

BG_PATH = Path(__file__).parent / "stadium.jpg"


def set_background():
    if not BG_PATH.exists():
        return
    b64 = base64.b64encode(BG_PATH.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
                url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===============================================================
# 데이터 (네이버 스포츠 저장 HTML에서 옮김)
# ===============================================================
HIT_COLS = ["시즌", "타율", "경기수", "타수", "안타", "2루타", "3루타", "홈런",
            "타점", "득점", "도루", "볼넷", "삼진", "출루율", "장타율", "OPS",
            "IsoP", "BABIP", "wOBA", "wRC+", "WPA", "WAR"]
PIT_COLS = ["시즌", "평균자책", "경기수", "이닝", "승", "패", "세이브", "홀드", "탈삼진",
            "피안타", "피홈런", "실점", "볼넷", "사구", "승률", "WHIP", "K/9", "BB/9",
            "K/BB", "K%", "BB%", "WPA", "WAR"]

GU = [
    [2015, .349, 116, 410, 143, 33, 5, 11, 57, 97, 17, 45, 79, .417, .534, .951, .185, .407, .420, 0.0, 1.07, 0.00],
    [2016, .343, 108, 428, 147, 19, 13, 14, 77, 105, 10, 55, 68, .420, .547, .967, .204, .379, .425, 0.0, 3.69, 0.00],
    [2017, .310, 144, 564, 175, 39, 10, 21, 107, 108, 10, 63, 138, .383, .527, .910, .217, .371, .399, 123.8, 1.65, 5.75],
    [2018, .333, 116, 478, 159, 26, 5, 20, 84, 100, 10, 43, 110, .392, .533, .925, .200, .397, .403, 117.8, 2.88, 5.22],
    [2019, .267, 122, 475, 127, 27, 6, 15, 71, 66, 11, 38, 88, .327, .444, .771, .177, .296, .347, 96.6, -0.26, 3.45],
    [2020, .307, 118, 446, 137, 27, 2, 15, 78, 70, 19, 51, 91, .385, .478, .863, .171, .355, .382, 105.0, 2.43, 4.18],
    [2021, .306, 139, 543, 166, 30, 10, 22, 88, 107, 27, 48, 98, .361, .519, .880, .213, .331, .385, 130.0, 1.25, 5.33],
    [2022, .293, 99, 409, 120, 23, 3, 5, 38, 69, 11, 27, 84, .340, .401, .741, .108, .357, .332, 101.0, -1.23, 1.86],
    [2023, .336, 119, 453, 152, 37, 1, 11, 71, 65, 12, 53, 81, .407, .494, .901, .158, .387, .396, 145.7, 1.87, 4.96],
    [2024, .343, 129, 493, 169, 39, 1, 33, 115, 92, 13, 55, 73, .417, .627, 1.044, .284, .346, .445, 151.6, 4.77, 6.17],
    [2025, .319, 142, 529, 169, 43, 2, 19, 96, 106, 4, 73, 91, .402, .516, .918, .197, .351, .406, 142.9, 3.04, 5.10],
    [2026, .371, 114, 420, 156, 31, 6, 15, 102, 88, 6, 66, 71, .458, .581, 1.039, .210, .416, .452, 167.8, 3.71, 6.14],
]
KIM = [
    [2022, .237, 103, 224, 53, 11, 4, 3, 19, 37, 13, 22, 62, .312, .362, .674, .125, .313, .307, 90.7, -1.63, 0.40],
    [2023, .303, 84, 340, 103, 20, 5, 7, 47, 72, 25, 38, 62, .371, .453, .824, .150, .349, .368, 133.3, -0.40, 2.74],
    [2024, .347, 141, 544, 189, 29, 10, 38, 109, 143, 40, 66, 110, .420, .647, 1.067, .300, .375, .454, 167.5, 4.24, 7.34],
    [2025, .309, 30, 110, 34, 9, 0, 7, 27, 20, 3, 10, 23, .361, .582, .943, .273, .329, .400, 149.9, 0.93, 1.04],
    [2026, .308, 135, 496, 153, 29, 2, 45, 118, 120, 11, 85, 103, .413, .647, 1.060, .339, .306, .447, 165.7, 5.82, 7.20],
]
AUS = [
    [2023, .313, 139, 520, 163, 29, 4, 23, 95, 87, 7, 53, 75, .376, .517, .893, .204, .326, .392, 158.5, 0.98, 5.57],
    [2024, .319, 140, 527, 168, 32, 3, 32, 132, 99, 12, 61, 82, .384, .573, .957, .254, .319, .408, 146.2, 1.35, 5.50],
    [2025, .313, 116, 425, 133, 25, 1, 31, 95, 82, 3, 61, 62, .393, .595, .988, .282, .297, .418, 168.4, 2.13, 5.67],
    [2026, .331, 144, 543, 180, 34, 4, 41, 127, 114, 5, 81, 95, .427, .635, 1.062, .304, .338, .453, 182.6, 6.72, 8.72],
]
AHN = [
    [2022, .133, 12, 15, 2, 0, 0, 0, 1, 2, 1, 3, 3, .188, .133, .321, .000, .167, .158, 0.0, 0.00, 0.00],
    [2024, .200, 16, 25, 5, 0, 1, 1, 2, 5, 0, 2, 10, .276, .400, .676, .200, .267, .290, 64.1, -0.05, 0.08],
    [2025, .334, 112, 395, 132, 19, 4, 22, 80, 72, 7, 75, 72, .448, .570, 1.018, .236, .362, .438, 172.5, 2.45, 7.22],
    [2026, .298, 94, 326, 97, 22, 2, 14, 63, 56, 5, 64, 90, .441, .506, .947, .208, .367, .423, 156.4, 1.93, 4.45],
]
DIAZ = [
    [2024, .282, 29, 110, 31, 5, 0, 7, 19, 14, 0, 6, 25, .331, .518, .849, .236, .308, .362, 101.7, 0.11, 0.44],
    [2025, .314, 144, 551, 173, 32, 0, 50, 158, 93, 1, 60, 100, .381, .644, 1.025, .330, .299, .426, 148.6, 3.11, 6.23],
    [2026, .305, 144, 541, 165, 33, 0, 26, 129, 86, 2, 73, 98, .384, .510, .894, .205, .326, .387, 121.7, 2.20, 4.17],
]
KWAK = [
    [2018, 7.55, 32, 31.0, 3, 1, 1, 4, 26, 44, 6, 26, 17, 3, .750, 1.97, 7.5, 4.9, 1.5, 16.9, 11.0, 0.28, -0.76],
    [2021, 4.10, 21, 98 + 2/3, 4, 7, 0, 0, 96, 78, 7, 50, 79, 12, .364, 1.59, 8.8, 7.2, 1.2, 21.1, 17.4, 0.77, 1.07],
    [2022, 3.78, 27, 147 + 2/3, 8, 9, 0, 0, 138, 143, 13, 69, 60, 11, .471, 1.37, 8.4, 3.7, 2.3, 21.4, 9.3, 1.61, 1.92],
    [2023, 2.90, 23, 127 + 1/3, 12, 7, 0, 0, 106, 96, 7, 44, 58, 5, .632, 1.21, 7.5, 4.1, 1.8, 20.2, 11.0, 4.03, 2.91],
    [2024, 4.24, 30, 167 + 2/3, 15, 9, 0, 0, 154, 142, 11, 83, 76, 6, .625, 1.30, 8.3, 4.1, 2.0, 21.5, 10.6, 2.40, 3.24],
    [2025, 4.20, 19, 109 + 1/3, 5, 7, 0, 0, 107, 96, 9, 55, 41, 2, .417, 1.25, 8.8, 3.4, 2.6, 23.0, 8.8, 1.38, 1.68],
    [2026, 2.24, 27, 161.0, 11, 7, 0, 0, 197, 129, 11, 45, 39, 5, .611, 1.04, 11.0, 2.2, 5.0, 30.4, 6.0, 4.59, 6.16],
]
RYU = [
    [2006, 2.23, 30, 201 + 2/3, 18, 6, 1, 0, 204, 159, 11, 57, 52, 2, .750, 1.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
    [2007, 2.94, 30, 211.0, 17, 7, 0, 0, 178, 195, 15, 74, 68, 3, .708, 1.25, 7.6, 2.9, 2.6, 20.5, 7.8, 0.00, 0.00],
    [2008, 3.31, 26, 165 + 2/3, 14, 7, 0, 0, 143, 144, 12, 66, 67, 1, .667, 1.27, 7.8, 3.6, 2.1, 20.9, 9.8, 0.00, 0.00],
    [2009, 3.57, 28, 189 + 1/3, 13, 12, 0, 0, 188, 180, 19, 80, 67, 2, .520, 1.30, 8.9, 3.2, 2.8, 23.7, 8.4, 0.00, 0.00],
    [2010, 1.82, 25, 192 + 2/3, 16, 4, 0, 0, 187, 149, 11, 42, 45, 9, .800, 1.01, 8.7, 2.1, 4.2, 24.7, 6.0, 0.00, 0.00],
    [2011, 3.36, 24, 126.0, 11, 7, 0, 0, 128, 101, 12, 54, 38, 1, .611, 1.10, 9.1, 2.7, 3.4, 24.6, 7.3, 0.00, 0.00],
    [2012, 2.66, 27, 182 + 2/3, 9, 9, 0, 0, 210, 153, 12, 58, 46, 5, .500, 1.09, 10.3, 2.3, 4.6, 28.6, 6.3, 0.00, 0.00],
    [2024, 3.87, 28, 158 + 1/3, 10, 8, 0, 0, 135, 182, 12, 78, 33, 3, .556, 1.36, 7.7, 1.9, 4.1, 19.9, 4.9, 2.23, 3.72],
    [2025, 3.23, 26, 139 + 1/3, 9, 7, 0, 0, 122, 144, 12, 54, 25, 4, .563, 1.21, 7.9, 1.6, 4.9, 21.3, 4.4, 3.15, 3.64],
    [2026, 3.88, 25, 134 + 2/3, 9, 6, 0, 0, 115, 136, 10, 68, 20, 6, .600, 1.16, 7.7, 1.3, 5.8, 20.5, 3.6, 2.50, 2.86],
]
CHOI = [
    [2025, 4.40, 17, 77 + 2/3, 3, 3, 0, 0, 53, 72, 7, 41, 34, 5, .500, 1.36, 6.1, 3.9, 1.6, 15.9, 10.2, 1.05, 0.68],
    [2026, 3.20, 27, 146 + 1/3, 14, 5, 0, 0, 129, 143, 8, 65, 54, 7, .737, 1.35, 7.9, 3.3, 2.4, 20.4, 8.5, 3.52, 3.14],
]
KO = [
    [2015, 5.68, 46, 57.0, 3, 4, 0, 0, 57, 68, 2, 41, 19, 16, .429, 1.53, 9.0, 3.0, 3.0, 21.4, 7.1, -1.11, 0.58],
    [2016, 5.59, 53, 56 + 1/3, 2, 4, 0, 5, 62, 65, 6, 38, 19, 10, .333, 1.49, 9.9, 3.0, 3.3, 23.8, 7.3, -1.06, 0.44],
    [2017, 5.08, 25, 141 + 2/3, 8, 12, 0, 1, 125, 170, 13, 84, 16, 18, .400, 1.31, 7.9, 1.0, 7.8, 20.4, 2.6, 0.85, 3.27],
    [2018, 5.13, 25, 142.0, 6, 9, 0, 0, 134, 175, 17, 91, 25, 14, .400, 1.41, 8.5, 1.6, 5.4, 21.6, 4.0, 0.45, 3.05],
    [2021, 2.92, 26, 166 + 2/3, 11, 6, 0, 1, 130, 147, 9, 59, 27, 14, .647, 1.04, 7.0, 1.5, 4.8, 19.6, 4.1, 4.12, 5.41],
    [2022, 3.26, 28, 182 + 1/3, 13, 8, 0, 0, 156, 191, 7, 69, 23, 16, .619, 1.17, 7.7, 1.1, 6.8, 20.6, 3.0, 4.30, 5.16],
    [2023, 2.78, 28, 174 + 2/3, 12, 7, 0, 0, 114, 181, 7, 57, 19, 9, .632, 1.15, 5.9, 1.0, 6.0, 16.1, 2.7, 5.26, 5.56],
    [2024, 4.95, 18, 100.0, 6, 8, 0, 0, 79, 141, 6, 61, 14, 11, .429, 1.55, 7.1, 1.3, 5.6, 17.3, 3.1, 0.67, 1.71],
    [2025, 3.30, 29, 161.0, 11, 8, 0, 0, 154, 170, 10, 70, 30, 15, .579, 1.24, 8.6, 1.7, 5.1, 22.5, 4.4, 2.96, 3.96],
    [2026, 3.28, 29, 172 + 2/3, 15, 7, 0, 0, 183, 164, 16, 65, 22, 11, .682, 1.08, 9.5, 1.1, 8.3, 25.9, 3.1, 3.18, 5.22],
]

# 존 차트 종류: (제목, 색상, 숫자형식)
ZK = {
    "hot": ("HOT & COLD ZONE (타율)", "coolwarm", "{:.3f}"),
    "k": ("삼진 분포도 (%)", "Reds", "{:.1f}%"),
    "pitch": ("투구 분포도 (%)", "Oranges", "{:.1f}%"),
    "avg": ("구역별 피안타율", "coolwarm_r", "{:.3f}"),
    "pk": ("탈삼진 분포도 (%)", "Reds", "{:.1f}%"),
}


def Z(kind, outer, inner9):
    title, cmap, fmt = ZK[kind]
    inner = [inner9[0:3], inner9[3:6], inner9[6:9]]
    return title, {"outer": outer, "inner": inner, "cmap": cmap, "fmt": fmt}


def hitter(team, pos, no, rows, rank, hot, k, field, vs, note=""):
    return {"유형": "타자", "팀": team, "포지션": pos, "등번호": no,
            "시즌기록": pd.DataFrame(rows, columns=HIT_COLS), "랭킹": rank,
            "존차트": dict([Z("hot", *hot), Z("k", *k)]),
            "타구분포": dict(zip(["레프트", "센터", "라이트"], field)),
            "VS팀": vs, "VS팀_라벨": "타율", "비고": note}


def pitcher(team, pos, no, rows, rank, pitch, avg, pk, kinds, vs, note=""):
    names, spd, lg, ratio = kinds
    return {"유형": "투수", "팀": team, "포지션": pos, "등번호": no,
            "시즌기록": pd.DataFrame(rows, columns=PIT_COLS), "랭킹": rank,
            "존차트": dict([Z("pitch", *pitch), Z("avg", *avg), Z("pk", *pk)]),
            "구종": pd.DataFrame({"구종": names, "선수 구속": spd,
                                "리그평균 구속": lg, "구사비율": ratio}),
            "VS팀": vs, "VS팀_라벨": "평균자책점", "비고": note}


PLAYERS = {
    "구자욱": hitter(
        "삼성 라이온즈", "외야수 (우투좌타)", 5, GU,
        {"타율": "1위", "OPS": "3위", "WAR": "6위", "타점": "8위", "득점": "공동 8위", "안타": "8위"},
        ([.455, .100, .375, .258], [.258, .310, .342, .481, .389, .364, .452, .477, .429]),
        ([2.8, 12.7, 15.5, 16.9], [2.8, 5.6, 15.5, 1.4, 2.8, 9.9, 7.0, 5.6, 1.4]),
        (36.3, 28.5, 35.2),
        {"한화": .576, "SSG": .457, "두산": .405, "LG": .392, "롯데": .344,
         "KIA": .340, "KT": .333, "NC": .288, "키움": .286}),
    "김도영": hitter(
        "KIA 타이거즈", "내야수 (우투우타)", 5, KIM,
        {"홈런": "1위", "득점": "2위", "OPS": "2위", "WAR": "3위", "타점": "공동 4위",
         "타율": "9위", "안타": "공동 11위"},
        ([.296, .083, .171, .059], [.282, .356, .333, .333, .500, .333, .264, .238, .429]),
        ([9.7, 9.7, 15.5, 13.6], [12.6, 4.9, 5.8, 2.9, 1.0, 1.0, 9.7, 7.8, 5.8]),
        (57.3, 24.1, 18.6),
        {"키움": .448, "KT": .368, "한화": .328, "롯데": .322, "삼성": .306,
         "두산": .255, "SSG": .255, "NC": .244, "LG": .208},
        "2025 시즌은 30경기만 출전해서 수치가 작아요."),
    "오스틴 딘": hitter(
        "LG 트윈스", "내야수 (우투우타)", 23, AUS,
        {"OPS": "1위", "WAR": "1위", "홈런": "3위", "타점": "3위", "안타": "3위",
         "타율": "4위", "득점": "4위"},
        ([.308, .219, .164, .364], [.333, .310, .409, .364, .302, .300, .358, .395, .473]),
        ([6.3, 6.3, 29.5, 4.2], [4.2, 4.2, 1.1, 5.3, 5.3, 5.3, 9.5, 7.4, 11.6]),
        (50.4, 22.8, 26.8),
        {"KT": .403, "키움": .385, "한화": .367, "SSG": .351, "두산": .348,
         "롯데": .339, "NC": .304, "삼성": .246, "KIA": .228},
        "KBO 입단 이후(2023~2026) 기록만 있어요."),
    "안현민": hitter(
        "KT 위즈", "외야수 (우투우타)", 23, AHN, {"WAR": "12위"},
        ([.333, .308, .162, .176], [.267, .185, .379, .367, .520, .333, .317, .387, .143]),
        ([6.7, 2.2, 16.7, 12.2], [8.9, 6.7, 3.3, 4.4, 0.0, 4.4, 12.2, 10.0, 12.2]),
        (50.0, 23.8, 26.3),
        {"한화": .391, "삼성": .300, "두산": .298, "LG": .290, "KIA": .286,
         "롯데": .283, "SSG": .273, "키움": .273, "NC": .243},
        "2023년 기록이 표에 없고, 2022·2024년은 경기 수가 적어요(일부 지표는 원본에서 0)."),
    "르윈 디아즈": hitter(
        "삼성 라이온즈", "내야수 (좌투좌타)", 0, DIAZ,
        {"타점": "2위", "안타": "6위", "홈런": "8위", "득점": "10위", "OPS": "11위",
         "타율": "12위", "WAR": "13위"},
        ([.188, .314, .226, .197], [.333, .262, .261, .421, .275, .472, .343, .418, .278]),
        ([6.1, 9.2, 15.3, 23.5], [4.1, 2.0, 8.2, 1.0, 2.0, 4.1, 4.1, 7.1, 13.3]),
        (27.9, 27.2, 44.9),
        {"NC": .373, "롯데": .349, "한화": .333, "SSG": .328, "KT": .310,
         "키움": .293, "LG": .279, "두산": .258, "KIA": .214},
        "2024년은 29경기만 출전했어요. 2024~2026 기록만 있어요."),
    "곽빈": pitcher(
        "두산 베어스", "투수 (우투우타)", 47, KWAK,
        {"평균자책": "1위", "탈삼진": "1위", "WHIP": "1위", "WAR": "1위", "다승": "공동 8위", "이닝": "8위"},
        ([9.5, 17.6, 13.9, 8.9], [5.9, 5.9, 7.8, 5.1, 5.8, 5.5, 4.8, 4.5, 5.0]),
        ([.094, .068, .145, .133], [.278, .264, .115, .333, .380, .209, .225, .324, .189]),
        ([7.1, 10.2, 13.7, 11.7], [6.6, 6.1, 10.7, 3.6, 5.1, 6.1, 7.1, 4.6, 7.6]),
        (["직구", "커터", "슬라이더", "커브", "체인지업"], [152, 146, 136, 121, 134],
         [145, 138, 132, 120, 129], [43.2, 17.5, 14.1, 12.8, 12.4]),
        {"KT": .75, "KIA": .95, "LG": 1.96, "롯데": 2.00, "SSG": 2.08,
         "삼성": 2.25, "한화": 2.30, "키움": 3.00, "NC": 7.20},
        "2019·2020 시즌 기록이 표에 없어서 그래프에서 해당 연도가 빠져 있어요."),
    "류현진": pitcher(
        "한화 이글스", "투수 (좌투우타)", 99, RYU,
        {"다승": "공동 14위", "WAR": "14위", "이닝": "20위"},
        ([12.0, 11.4, 9.9, 12.9], [6.0, 5.4, 7.9, 5.7, 5.7, 5.9, 6.0, 4.2, 6.9]),
        ([.111, .217, .088, .143], [.333, .400, .289, .268, .356, .348, .375, .214, .222]),
        ([10.4, 6.1, 20.9, 10.4], [5.2, 5.2, 7.8, 5.2, 0.0, 4.3, 7.0, 4.3, 13.0]),
        (["직구", "체인지업", "커터", "커브", "스위퍼", "투심"], [142, 127, 135, 111, 123, 139],
         [145, 129, 138, 120, 128, 143], [44.0, 24.5, 17.4, 10.8, 2.8, 0.4]),
        {"SSG": 2.60, "롯데": 2.81, "KIA": 3.00, "KT": 3.42, "NC": 4.38,
         "두산": 4.91, "키움": 6.10, "LG": 18.00},
        "KBO 기록만 있어요. 2013~2023년은 표에 없어서 점으로만 표시돼요. 2006년 일부 지표는 원본에서 0이에요."),
    "최민석": pitcher(
        "두산 베어스", "투수 (우투우타)", 68, CHOI,
        {"평균자책": "3위", "다승": "3위", "WAR": "10위", "탈삼진": "13위", "WHIP": "13위", "이닝": "15위"},
        ([11.0, 13.2, 17.2, 5.1], [8.1, 6.2, 6.9, 7.3, 5.0, 4.6, 7.3, 4.2, 3.9]),
        ([.190, .139, .111, .261], [.169, .267, .222, .224, .346, .350, .125, .359, .564]),
        ([3.9, 17.1, 12.4, 3.9], [16.3, 1.6, 9.3, 12.4, 3.9, 0.8, 14.7, 2.3, 1.6]),
        (["투심", "커터", "스위퍼", "포크", "슬라이더"], [143, 139, 127, 136, 130],
         [143, 138, 128, 132, 132], [46.8, 27.3, 11.9, 11.7, 2.2]),
        {"롯데": 0.00, "한화": 1.64, "KIA": 2.08, "키움": 2.20, "NC": 2.77,
         "KT": 3.71, "SSG": 3.86, "삼성": 5.68, "LG": 5.74},
        "시즌이 2개(2025, 2026)뿐이라 추세를 해석하기엔 자료가 적어요."),
    "고영표": pitcher(
        "kt wiz", "투수 (우언우타)", 1, KO,
        {"다승": "공동 1위", "WHIP": "2위", "WAR": "2위", "탈삼진": "3위", "이닝": "3위", "평균자책": "4위"},
        ([7.4, 7.5, 10.8, 21.0], [5.9, 5.2, 6.5, 5.6, 5.6, 6.0, 5.8, 5.6, 7.3]),
        ([.000, .176, .156, .147], [.321, .341, .226, .231, .271, .350, .341, .333, .288]),
        ([3.8, 3.8, 18.6, 30.6], [4.9, 3.8, 9.3, 2.2, 1.6, 1.6, 5.5, 6.0, 8.2]),
        (["체인지업", "투심", "스위퍼", "커브", "직구", "커터", "슬라이더"], [119, 135, 118, 116, 133, 127, 123],
         [129, 143, 128, 120, 145, 138, 132], [41.8, 35.0, 12.3, 5.9, 4.9, 0.1, 0.0]),
        {"롯데": 1.40, "두산": 2.08, "삼성": 2.25, "LG": 2.37, "KIA": 3.00,
         "SSG": 3.27, "한화": 4.18, "키움": 5.24, "NC": 5.82},
        "2019·2020 시즌과 2014년 이전 기록이 표에 없어요. 구속이 모두 리그 평균보다 느린 '완급 조절형' 투수예요."),
}


# ===============================================================
# 그래프 함수
# ===============================================================
def zone_heatmap(d, title):
    outer, inner = d["outer"], d["inner"]
    vals = outer + [v for row in inner for v in row]
    norm = plt.Normalize(min(vals), max(vals))
    cm = plt.get_cmap(d["cmap"])
    fmt = d["fmt"]
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = [(0, 2), (2, 2), (0, 0), (2, 0)]
    for (x, y), v in zip(pos, outer):
        ax.add_patch(Rectangle((x, y), 2, 2, facecolor=cm(norm(v)), edgecolor="white", lw=2))
        tx = x + 0.3 if x == 0 else x + 1.7
        ty = y + 1.7 if y == 2 else y + 0.3
        ax.text(tx, ty, fmt.format(v), ha="left" if x == 0 else "right",
                va="center", fontsize=11, fontweight="bold")
    cell = 2 / 3
    start = 2 - 1.5 * cell
    for i in range(3):
        for j in range(3):
            v = inner[i][j]
            x, y = start + j * cell, start + (2 - i) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=cm(norm(v)),
                                   edgecolor="black", lw=1))
            ax.text(x + cell / 2, y + cell / 2, fmt.format(v), ha="center",
                    va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "\n(투수 시점)", fontsize=13)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


def field_chart(d):
    left, center, right = d["레프트"], d["센터"], d["라이트"]
    norm = plt.Normalize(0, max(left, center, right) * 1.2)
    cm = plt.get_cmap("YlOrRd")
    fig, ax = plt.subplots(figsize=(5, 5))
    for nm, a1, a2, v in [("레프트", 105, 135, left), ("센터", 75, 105, center),
                          ("라이트", 45, 75, right)]:
        ax.add_patch(Wedge((0, 0), 10, a1, a2, facecolor=cm(norm(v)), edgecolor="white", lw=2))
        mid = np.deg2rad((a1 + a2) / 2)
        ax.text(6.5 * np.cos(mid), 6.5 * np.sin(mid), f"{nm}\n{v:.1f}%",
                ha="center", va="center", fontsize=12, fontweight="bold")
    ax.plot(0, 0, "ko", markersize=8)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-1, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("타구 분포도", fontsize=13)
    fig.tight_layout()
    return fig


def pitch_speed_chart(df, name):
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, df["선수 구속"], w, label=name, color="#f46d46")
    b2 = ax.bar(x + w / 2, df["리그평균 구속"], w, label="리그평균", color="#77a4f0")
    ax.bar_label(b1, fmt="%d")
    ax.bar_label(b2, fmt="%d")
    ax.set_xticks(x)
    ax.set_xticklabels(df["구종"])
    ax.set_ylim(100, max(df["리그평균 구속"].max(), df["선수 구속"].max()) + 10)
    ax.set_ylabel("km/h")
    ax.set_title(f"{name} 구종별 구속 vs 리그평균")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def pitch_ratio_chart(df, name):
    d = df[df["구사비율"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(d["구사비율"], labels=d["구종"], autopct="%1.1f%%", startangle=90,
           counterclock=False, wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.set_title(f"{name} 구사비율")
    fig.tight_layout()
    return fig


def trend_chart(df, metrics, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    has_gap = df["시즌"].diff().max() > 1
    for m in metrics:
        ax.plot(df["시즌"], df[m], marker="o",
                linestyle="None" if has_gap else "-", label=m)
    ax.set_xticks(df["시즌"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def vs_team_chart(data, name, label):
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([k for k, _ in items], [v for _, v in items], color="#77a4f0")
    ax.bar_label(bars, fmt="%.3f" if label == "타율" else "%.2f")
    ax.set_title(f"{name} 팀별 {label}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def compare_chart(names, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    seasons = set()
    for n in names:
        d = PLAYERS[n]["시즌기록"]
        if metric in d.columns:
            ax.plot(d["시즌"], d[metric], marker="o", label=n)
            seasons |= set(d["시즌"])
    ax.set_xticks(sorted(seasons))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{metric} 비교")
    fig.tight_layout()
    return fig


# ===============================================================
# 화면 (선수 기록실)
# ===============================================================
set_background()
st.title("⚾ KBO 선수 기록실")
st.caption("선수 이름을 입력하면 등록된 실제 기록을 그래프로 보여줍니다. (데이터 출처: 네이버 스포츠 선수 기록 페이지)")

query = st.text_input("선수 이름", placeholder="예: 곽빈").strip()
st.write("등록된 선수:", ", ".join(f"{n}({p['유형']})" for n, p in PLAYERS.items()))

if query:
    matches = [n for n in PLAYERS if query in n]
    if not matches:
        st.warning(f"'{query}' 선수는 아직 등록되지 않았어요. "
                   "그 선수의 기록 페이지를 저장한 HTML을 보내주시면 추가할 수 있어요.")
    else:
        name = matches[0] if len(matches) == 1 else st.selectbox("선수 선택", matches)
        p = PLAYERS[name]
        df = p["시즌기록"]
        is_pitcher = p["유형"] == "투수"
        latest = df.iloc[-1]
        season = int(latest["시즌"])

        st.header(f"{name}  NO.{p['등번호']}")
        st.write(f"{p['팀']} | {p['포지션']}")

        c = st.columns(5)
        if is_pitcher:
            c[0].metric(f"{season} 평균자책", f"{latest['평균자책']:.2f}")
            c[1].metric("승-패", f"{int(latest['승'])}-{int(latest['패'])}")
            c[2].metric("이닝", f"{latest['이닝']:.1f}")
            c[3].metric("탈삼진", int(latest["탈삼진"]))
            c[4].metric("WHIP", f"{latest['WHIP']:.2f}")
        else:
            c[0].metric(f"{season} 타율", f"{latest['타율']:.3f}")
            c[1].metric("홈런", int(latest["홈런"]))
            c[2].metric("타점", int(latest["타점"]))
            c[3].metric("OPS", f"{latest['OPS']:.3f}")
            c[4].metric("WAR", f"{latest['WAR']:.2f}")

        st.subheader("🏆 시즌 랭킹")
        st.write(" · ".join(f"**{k}** {v}" for k, v in p["랭킹"].items()))

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔥 존 차트", "🎯 구종·팀별" if is_pitcher else "⚾ 타구·팀별", "📈 시즌 추이", "📄 표/CSV"])

        with tab1:
            charts = list(p["존차트"].items())
            cols = st.columns(len(charts))
            for col, (title, d) in zip(cols, charts):
                with col:
                    st.pyplot(zone_heatmap(d, f"{name} {title}"))
            if is_pitcher:
                st.caption("※ 구역별 피안타율은 낮을수록 투수에게 유리해서 낮은 칸을 빨간색으로 표시했어요.")

        with tab2:
            a, b = st.columns(2)
            if is_pitcher:
                with a:
                    st.pyplot(pitch_speed_chart(p["구종"], name))
                with b:
                    st.pyplot(pitch_ratio_chart(p["구종"], name))
                st.pyplot(vs_team_chart(p["VS팀"], name, p["VS팀_라벨"]))
                st.caption("평균자책점은 낮을수록 좋아요. 이닝이 적은 상대의 값은 믿기 어려워요.")
            else:
                with a:
                    st.pyplot(field_chart(p["타구분포"]))
                with b:
                    st.pyplot(vs_team_chart(p["VS팀"], name, p["VS팀_라벨"]))

        with tab3:
            cols_ = [c_ for c_ in df.columns if c_ != "시즌"]
            default = ["평균자책", "WHIP"] if is_pitcher else ["타율", "출루율", "장타율"]
            chosen = st.multiselect("항목 선택", cols_, default=default, key=f"trend_{name}")
            if chosen:
                st.pyplot(trend_chart(df, chosen, f"{name} 시즌별 기록"))
            if p["비고"]:
                st.caption("※ " + p["비고"])
            if not is_pitcher:
                st.caption("※ 값의 크기가 많이 다른 항목을 같이 고르면 한쪽이 납작해 보여요.")

        with tab4:
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ 시즌 기록 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{name}_시즌기록.csv", mime="text/csv")

st.divider()
st.subheader("🆚 선수 비교")
sel = st.multiselect("비교할 선수", list(PLAYERS.keys()), default=["구자욱", "김도영", "오스틴 딘"])
if sel:
    common = None
    for n in sel:
        cols_ = set(PLAYERS[n]["시즌기록"].columns) - {"시즌"}
        common = cols_ if common is None else common & cols_
    common = sorted(common)
    if common:
        metric = st.selectbox("항목", common, index=common.index("WAR") if "WAR" in common else 0)
        st.pyplot(compare_chart(sel, metric))
    else:
        st.info("선택한 선수들의 공통 항목이 없어요. 타자끼리, 투수끼리 비교하거나 WAR 같은 공통 항목을 이용하세요.")


# ===============================================================
# 🤖 군집 모델 평가 및 선택
# ===============================================================
MODEL_NAMES = ["K-Means", "계층적 군집", "가우시안 혼합(GMM)"]


@st.cache_data
def build_cluster_data(player_type):
    """선수 유형(타자/투수)별로 '선수-시즌' 한 행짜리 데이터프레임을 만든다."""
    frames = []
    for n, info in PLAYERS.items():
        if info["유형"] != player_type:
            continue
        d = info["시즌기록"].copy()
        d.insert(0, "선수", n)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["라벨"] = data["선수"] + " " + data["시즌"].astype(str)
    return data


def fit_labels(model_name, X, k, linkage="ward", cov_type="full"):
    if model_name == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    if model_name == "계층적 군집":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    return GaussianMixture(n_components=k, covariance_type=cov_type,
                           random_state=42).fit(X).predict(X)


def run_clustering_section():
    st.divider()
    st.header("🤖 군집 모델 평가 및 선택")
    st.caption("선수-시즌 기록을 비슷한 유형끼리 묶는 군집 모델 3가지를 비교하고, "
               "가장 적합한 모델을 선택해 보세요.")

    # ---------- 설정 ----------
    left, _ = st.columns([1, 2])
    with left:
        ptype = st.radio("분석 대상", ["타자", "투수"], horizontal=True, key="cl_type")
        cdata = build_cluster_


# ===============================================================
# 🤖 군집 모델 평가 및 선택
# ===============================================================
from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["K-Means", "계층적 군집", "가우시안 혼합(GMM)"]


@st.cache_data
def build_cluster_data(player_type):
    frames = []
    for n, info in PLAYERS.items():
        if info["유형"] != player_type:
            continue
        d = info["시즌기록"].copy()
        d.insert(0, "선수", n)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["라벨"] = data["선수"] + " " + data["시즌"].astype(str)
    return data


def fit_labels(model_name, X, k, linkage="ward", cov_type="full"):
    if model_name == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    if model_name == "계층적 군집":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    return GaussianMixture(n_components=k, covariance_type=cov_type,
                           random_state=42).fit(X).predict(X)


def run_clustering_section():
    st.divider()
    st.header("🤖 군집 모델 평가 및 선택")
    st.caption("선수-시즌 기록을 비슷한 유형끼리 묶는 군집 모델 3가지를 비교하고, "
               "가장 적합한 모델을 선택해 보세요.")

    # ---------- 설정 ----------
    left, _ = st.columns([1, 2])
    with left:
        ptype = st.radio("분석 대상", ["타자", "투수"], horizontal=True, key="cl_type")
        cdata = build_cluster_data(ptype)

        drop_zero = st.checkbox("WAR·wRC+ 등이 0인(원본 결측 의심) 행 제외", value=True)
        if drop_zero:
            if ptype == "타자":
                cdata = cdata[(cdata["wRC+"] > 0) & (cdata["경기수"] >= 30)]
            else:
                cdata = cdata[(cdata["WAR"] != 0) & (cdata["K/9"] > 0)]
        cdata = cdata.reset_index(drop=True)

        num_cols = [c_ for c_ in cdata.columns if c_ not in ("선수", "시즌", "라벨")]
        default_feat = (["타율", "출루율", "장타율", "홈런", "도루", "WAR"]
                        if ptype == "타자"
                        else ["평균자책", "WHIP", "K/9", "BB/9", "WAR"])
        features = st.multiselect("군집에 사용할 특성", num_cols, default=default_feat,
                                  key=f"cl_feat_{ptype}")
        k_range = st.slider("탐색할 군집 수(K) 범위", 2, 8, (2, 6))
        k_final = st.slider("최종 확인할 군집 수(K)", 2, 8, 3)

    if len(features) < 2:
        st.info("특성을 2개 이상 선택하세요.")
        return
    if len(cdata) <= k_range[1]:
        st.warning("데이터 수가 너무 적어요. K 범위를 줄이거나 제외 옵션을 해제해 보세요.")
        return

    X = StandardScaler().fit_transform(cdata[features])
    st.write(f"사용 데이터: **{len(cdata)}행** (선수-시즌), 특성 **{len(features)}개** (표준화 적용)")

    # ---------- 1) 모델별 평가 ----------
    st.subheader("1️⃣ 모델별 성능 비교")
    rows = []
    for mname in MODEL_NAMES:
        for k in range(k_range[0], k_range[1] + 1):
            try:
                lab = fit_labels(mname, X, k)
                if len(set(lab)) < 2:
                    continue
                rows.append({
                    "모델": mname, "K": k,
                    "실루엣(↑)": silhouette_score(X, lab),
                    "Davies-Bouldin(↓)": davies_bouldin_score(X, lab),
                    "Calinski-Harabasz(↑)": calinski_harabasz_score(X, lab),
                })
            except Exception:
                continue
    res = pd.DataFrame(rows)
    if res.empty:
        st.error("평가 결과를 계산하지 못했어요. 설정을 바꿔 보세요.")
        return

    st.dataframe(
        res.style.format({"실루엣(↑)": "{:.3f}", "Davies-Bouldin(↓)": "{:.3f}",
                          "Calinski-Harabasz(↑)": "{:.1f}"}),
        use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["실루엣(↑)", "Davies-Bouldin(↓)", "Calinski-Harabasz(↑)"]):
        for mname in MODEL_NAMES:
            sub = res[res["모델"] == mname]
            ax.plot(sub["K"], sub[m], marker="o", label=mname)
        ax.set_title(m)
        ax.set_xlabel("군집 수 K")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    st.pyplot(fig)

    best = res.sort_values("실루엣(↑)", ascending=False).iloc[0]
    st.success(f"📌 실루엣 계수 기준 추천: **{best['모델']} (K={int(best['K'])})**, "
               f"실루엣 = {best['실루엣(↑)']:.3f}")
    st.caption("※ 지표마다 '좋다'고 하는 모델이 다를 수 있어요. 시각화와 군집의 의미까지 함께 판단해 보세요.")

    # ---------- 2) 모델 선택 & 시각화 ----------
    st.subheader("2️⃣ 모델 선택 및 군집 확인")
    chosen_model = st.selectbox("사용할 모델 선택", MODEL_NAMES, key="cl_model")
    linkage, cov_type = "ward", "full"
    if chosen_model == "계층적 군집":
        linkage = st.selectbox("연결 방식(linkage)", ["ward", "complete", "average"])
    if chosen_model == "가우시안 혼합(GMM)":
        cov_type = st.selectbox("공분산 유형", ["full", "tied", "diag", "spherical"])

    labels = fit_labels(chosen_model, X, k_final, linkage, cov_type)
    cdata["군집"] = labels

    if len(set(labels)) > 1:
        m1, m2, m3 = st.columns(3)
        m1.metric("실루엣", f"{silhouette_score(X, labels):.3f}")
        m2.metric("Davies-Bouldin", f"{davies_bouldin_score(X, labels):.3f}")
        m3.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X, labels):.1f}")

    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(X)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=70, edgecolor="white")
    if len(cdata) <= 70:
        for (px, py), t in zip(pts, cdata["라벨"]):
            ax2.annotate(t, (px, py), fontsize=7, alpha=0.8)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax2.set_title(f"{chosen_model} (K={k_final}) 군집 결과 - PCA 2차원")
    ax2.grid(alpha=0.3)
    ax2.legend(*sc.legend_elements(), title="군집")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("**군집별 특성 평균** (군집의 의미를 해석해 보세요)")
    profile = cdata.groupby("군집")[features].mean()
    profile["데이터 수"] = cdata.groupby("군집").size()
    st.dataframe(profile.style.format("{:.3f}", subset=features), use_container_width=True)

    with st.expander("군집별 구성원 보기"):
        for g in sorted(cdata["군집"].unique()):
            members = cdata.loc[cdata["군집"] == g, "라벨"].tolist()
            st.write(f"**군집 {g}** ({len(members)}개): " + ", ".join(members))

    if chosen_model == "계층적 군집":
        Z_mat = sch_linkage(X, method=linkage)
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        dendrogram(Z_mat, labels=cdata["라벨"].tolist(), leaf_rotation=90,
                   leaf_font_size=7, ax=ax3)
        ax3.set_title("덴드로그램")
        fig3.tight_layout()
        st.pyplot(fig3)

    # ---------- 3) 최종 선택 ----------
    st.subheader("3️⃣ 최종 모델 확정")
    if st.button("✅ 이 모델로 확정하기"):
        st.session_state["final_cluster_model"] = {
            "대상": ptype, "모델": chosen_model, "K": k_final, "특성": features}
    if "final_cluster_model" in st.session_state:
        f = st.session_state["final_cluster_model"]
        st.success(f"확정된 모델: [{f['대상']}] {f['모델']} / K={f['K']} / "
                   f"특성: {', '.join(f['특성'])}")

    st.download_button("⬇️ 군집 결과 CSV", cdata.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{ptype}_군집결과.csv", mime="text/csv")


# ===============================================================
# 🤖 군집 모델 평가 및 선택 (Plotly 버전)
# ===============================================================
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage as sch_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["K-Means", "계층적 군집", "가우시안 혼합(GMM)"]
MODEL_COLORS = {"K-Means": "#636EFA", "계층적 군집": "#EF553B",
                "가우시안 혼합(GMM)": "#00CC96"}


@st.cache_data
def build_cluster_data(player_type):
    frames = []
    for n, info in PLAYERS.items():
        if info["유형"] != player_type:
            continue
        d = info["시즌기록"].copy()
        d.insert(0, "선수", n)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["라벨"] = data["선수"] + " " + data["시즌"].astype(str)
    return data


def fit_labels(model_name, X, k, linkage="ward", cov_type="full"):
    if model_name == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    if model_name == "계층적 군집":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    return GaussianMixture(n_components=k, covariance_type=cov_type,
                           random_state=42).fit(X).predict(X)


def metric_figure(res):
    """3개 평가 지표를 한 화면에 나란히 보여주는 Plotly 그래프"""
    metrics = ["실루엣(↑)", "Davies-Bouldin(↓)", "Calinski-Harabasz(↑)"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
    for i, m in enumerate(metrics, start=1):
        for mname in MODEL_NAMES:
            sub = res[res["모델"] == mname]
            fig.add_trace(
                go.Scatter(x=sub["K"], y=sub[m], mode="lines+markers", name=mname,
                           line=dict(color=MODEL_COLORS[mname]),
                           legendgroup=mname, showlegend=(i == 1),
                           hovertemplate="K=%{x}<br>" + m + "=%{y:.3f}<extra>" + mname + "</extra>"),
                row=1, col=i)
        fig.update_xaxes(title_text="군집 수 K", dtick=1, row=1, col=i)
    fig.update_layout(height=380, template="plotly_white",
                      legend=dict(orientation="h", y=-0.25),
                      margin=dict(t=50, b=40))
    return fig


def pca_figure(cdata, X, labels, title):
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(X)
    plot_df = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1],
                            "군집": [f"군집 {g}" for g in labels],
                            "라벨": cdata["라벨"]})
    plot_df = plot_df.sort_values("군집")
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="군집", hover_name="라벨",
                     text="라벨" if len(plot_df) <= 40 else None,
                     template="plotly_white")
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")),
                      textposition="top center", textfont=dict(size=10))
    fig.update_layout(
        title=title, height=550,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    return fig


def profile_heatmap(cdata, features, labels):
    """군집별 평균(표준화 값)을 히트맵으로 표시: 빨강=평균보다 높음, 파랑=낮음"""
    z = pd.DataFrame(StandardScaler().fit_transform(cdata[features]),
                     columns=features)
    z["군집"] = [f"군집 {g}" for g in labels]
    prof = z.groupby("군집")[features].mean()
    fig = px.imshow(prof, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                    labels=dict(color="표준화 평균"))
    fig.update_layout(height=120 + 70 * len(prof), template="plotly_white",
                      title="군집별 특성 프로필 (전체 평균 대비)")
    return fig


def run_clustering_section():
    st.divider()
    st.header("🤖 군집 모델 평가 및 선택")
    st.caption("선수-시즌 기록을 비슷한 유형끼리 묶는 군집 모델 3가지를 비교하고, "
               "가장 적합한 모델을 선택해 보세요.")

    # ---------- 설정 ----------
    left, _ = st.columns([1, 2])
    with left:
        ptype = st.radio("분석 대상", ["타자", "투수"], horizontal=True, key="cl_type")
        cdata = build_cluster_data(ptype)

        drop_zero = st.checkbox("WAR·wRC+ 등이 0인(원본 결측 의심) 행 제외", value=True)
        if drop_zero:
            if ptype == "타자":
                cdata = cdata[(cdata["wRC+"] > 0) & (cdata["경기수"] >= 30)]
            else:
                cdata = cdata[(cdata["WAR"] != 0) & (cdata["K/9"] > 0)]
        cdata = cdata.reset_index(drop=True)

        num_cols = [c_ for c_ in cdata.columns if c_ not in ("선수", "시즌", "라벨")]
        default_feat = (["타율", "출루율", "장타율", "홈런", "도루", "WAR"]
                        if ptype == "타자"
                        else ["평균자책", "WHIP", "K/9", "BB/9", "WAR"])
        features = st.multiselect("군집에 사용할 특성", num_cols, default=default_feat,
                                  key=f"cl_feat_{ptype}")
        k_range = st.slider("탐색할 군집 수(K) 범위", 2, 8, (2, 6))
        k_final = st.slider("최종 확인할 군집 수(K)", 2, 8, 3)

    if len(features) < 2:
        st.info("특성을 2개 이상 선택하세요.")
        return
    if len(cdata) <= k_range[1]:
        st.warning("데이터 수가 너무 적어요. K 범위를 줄이거나 제외 옵션을 해제해 보세요.")
        return

    X = StandardScaler().fit_transform(cdata[features])
    st.write(f"사용 데이터: **{len(cdata)}행** (선수-시즌), 특성 **{len(features)}개** (표준화 적용)")

    # ---------- 1) 모델별 평가 ----------
    st.subheader("1️⃣ 모델별 성능 비교")
    rows = []
    for mname in MODEL_NAMES:
        for k in range(k_range[0], k_range[1] + 1):
            try:
                lab = fit_labels(mname, X, k)
                if len(set(lab)) < 2:
                    continue
                rows.append({
                    "모델": mname, "K": k,
                    "실루엣(↑)": silhouette_score(X, lab),
                    "Davies-Bouldin(↓)": davies_bouldin_score(X, lab),
                    "Calinski-Harabasz(↑)": calinski_harabasz_score(X, lab),
                })
            except Exception:
                continue
    res = pd.DataFrame(rows)
    if res.empty:
        st.error("평가 결과를 계산하지 못했어요. 설정을 바꿔 보세요.")
        return

    st.plotly_chart(metric_figure(res), use_container_width=True)
    with st.expander("📋 평가 점수 표 보기"):
        st.dataframe(
            res.style.format({"실루엣(↑)": "{:.3f}", "Davies-Bouldin(↓)": "{:.3f}",
                              "Calinski-Harabasz(↑)": "{:.1f}"}),
            use_container_width=True)

    best = res.sort_values("실루엣(↑)", ascending=False).iloc[0]
    st.success(f"📌 실루엣 계수 기준 추천: **{best['모델']} (K={int(best['K'])})**, "
               f"실루엣 = {best['실루엣(↑)']:.3f}")
    st.caption("※ 지표마다 '좋다'고 하는 모델이 다를 수 있어요. 시각화와 군집의 의미까지 함께 판단해 보세요.")

    # ---------- 2) 모델 선택 & 시각화 ----------
    st.subheader("2️⃣ 모델 선택 및 군집 확인")
    chosen_model = st.selectbox("사용할 모델 선택", MODEL_NAMES, key="cl_model")
    linkage, cov_type = "ward", "full"
    if chosen_model == "계층적 군집":
        linkage = st.selectbox("연결 방식(linkage)", ["ward", "complete", "average"])
    if chosen_model == "가우시안 혼합(GMM)":
        cov_type = st.selectbox("공분산 유형", ["full", "tied", "diag", "spherical"])

    labels = fit_labels(chosen_model, X, k_final, linkage, cov_type)
    cdata["군집"] = labels

    if len(set(labels)) > 1:
        m1, m2, m3 = st.columns(3)
        m1.metric("실루엣", f"{silhouette_score(X, labels):.3f}")
        m2.metric("Davies-Bouldin", f"{davies_bouldin_score(X, labels):.3f}")
        m3.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X, labels):.1f}")

    st.plotly_chart(
        pca_figure(cdata, X, labels, f"{chosen_model} (K={k_final}) 군집 결과 - PCA 2차원"),
        use_container_width=True)

    st.plotly_chart(profile_heatmap(cdata, features, labels), use_container_width=True)

    st.markdown("**군집별 특성 평균 (원래 단위)**")
    profile = cdata.groupby("군집")[features].mean()
    profile["데이터 수"] = cdata.groupby("군집").size()
    st.dataframe(profile.style.format("{:.3f}", subset=features), use_container_width=True)

    with st.expander("군집별 구성원 보기"):
        for g in sorted(cdata["군집"].unique()):
            members = cdata.loc[cdata["군집"] == g, "라벨"].tolist()
            st.write(f"**군집 {g}** ({len(members)}개): " + ", ".join(members))

    if chosen_model == "계층적 군집":
        dendro = ff.create_dendrogram(
            X, labels=cdata["라벨"].tolist(),
            linkagefun=lambda x: sch_linkage(x, method=linkage))
        dendro.update_layout(height=500, title="덴드로그램", template="plotly_white",
                             margin=dict(b=140))
        st.plotly_chart(dendro, use_container_width=True)

    # ---------- 3) 최종 선택 ----------
    st.subheader("3️⃣ 최종 모델 확정")
    if st.button("✅ 이 모델로 확정하기"):
        st.session_state["final_cluster_model"] = {
            "대상": ptype, "모델": chosen_model, "K": k_final, "특성": features}
    if "final_cluster_model" in st.session_state:
        f = st.session_state["final_cluster_model"]
        st.success(f"확정된 모델: [{f['대상']}] {f['모델']} / K={f['K']} / "
                   f"특성: {', '.join(f['특성'])}")

    st.download_button("⬇️ 군집 결과 CSV", cdata.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{ptype}_군집결과.csv", mime="text/csv")


# 반드시 마지막에 호출해야 화면에 나타납니다!
run_clustering_section()

# 반드시 마지막에 호출해야 화면에 나타납니다!
run_clustering_section()
