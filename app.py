import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle, Wedge

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

# ===============================================================
# 실제 데이터 (네이버 스포츠 선수 기록 페이지 저장본에서 옮김)
# ===============================================================
HIT_COLS = ["시즌", "타율", "경기수", "타수", "안타", "2루타", "3루타", "홈런",
            "타점", "득점", "도루", "볼넷", "삼진", "출루율", "장타율", "OPS",
            "IsoP", "BABIP", "wOBA", "wRC+", "WPA", "WAR"]

PIT_COLS = ["시즌", "평균자책", "경기수", "이닝", "승", "패", "세이브", "홀드", "탈삼진",
            "피안타", "피홈런", "실점", "볼넷", "사구", "승률", "WHIP", "K/9", "BB/9",
            "K/BB", "K%", "BB%", "WPA", "WAR"]

GU_ROWS = [
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

KIM_ROWS = [
    [2022, .237, 103, 224, 53, 11, 4, 3, 19, 37, 13, 22, 62, .312, .362, .674, .125, .313, .307, 90.7, -1.63, 0.40],
    [2023, .303, 84, 340, 103, 20, 5, 7, 47, 72, 25, 38, 62, .371, .453, .824, .150, .349, .368, 133.3, -0.40, 2.74],
    [2024, .347, 141, 544, 189, 29, 10, 38, 109, 143, 40, 66, 110, .420, .647, 1.067, .300, .375, .454, 167.5, 4.24, 7.34],
    [2025, .309, 30, 110, 34, 9, 0, 7, 27, 20, 3, 10, 23, .361, .582, .943, .273, .329, .400, 149.9, 0.93, 1.04],
    [2026, .308, 135, 496, 153, 29, 2, 45, 118, 120, 11, 85, 103, .413, .647, 1.060, .339, .306, .447, 165.7, 5.82, 7.20],
]

AUSTIN_ROWS = [
    [2023, .313, 139, 520, 163, 29, 4, 23, 95, 87, 7, 53, 75, .376, .517, .893, .204, .326, .392, 158.5, 0.98, 5.57],
    [2024, .319, 140, 527, 168, 32, 3, 32, 132, 99, 12, 61, 82, .384, .573, .957, .254, .319, .408, 146.2, 1.35, 5.50],
    [2025, .313, 116, 425, 133, 25, 1, 31, 95, 82, 3, 61, 62, .393, .595, .988, .282, .297, .418, 168.4, 2.13, 5.67],
    [2026, .331, 144, 543, 180, 34, 4, 41, 127, 114, 5, 81, 95, .427, .635, 1.062, .304, .338, .453, 182.6, 6.72, 8.72],
]

# 이닝은 소수로 변환 (109 1/3 -> 109.333)
KWAK_ROWS = [
    [2018, 7.55, 32, 31.0, 3, 1, 1, 4, 26, 44, 6, 26, 17, 3, .750, 1.97, 7.5, 4.9, 1.5, 16.9, 11.0, 0.28, -0.76],
    [2021, 4.10, 21, 98 + 2/3, 4, 7, 0, 0, 96, 78, 7, 50, 79, 12, .364, 1.59, 8.8, 7.2, 1.2, 21.1, 17.4, 0.77, 1.07],
    [2022, 3.78, 27, 147 + 2/3, 8, 9, 0, 0, 138, 143, 13, 69, 60, 11, .471, 1.37, 8.4, 3.7, 2.3, 21.4, 9.3, 1.61, 1.92],
    [2023, 2.90, 23, 127 + 1/3, 12, 7, 0, 0, 106, 96, 7, 44, 58, 5, .632, 1.21, 7.5, 4.1, 1.8, 20.2, 11.0, 4.03, 2.91],
    [2024, 4.24, 30, 167 + 2/3, 15, 9, 0, 0, 154, 142, 11, 83, 76, 6, .625, 1.30, 8.3, 4.1, 2.0, 21.5, 10.6, 2.40, 3.24],
    [2025, 4.20, 19, 109 + 1/3, 5, 7, 0, 0, 107, 96, 9, 55, 41, 2, .417, 1.25, 8.8, 3.4, 2.6, 23.0, 8.8, 1.38, 1.68],
    [2026, 2.24, 27, 161.0, 11, 7, 0, 0, 197, 129, 11, 45, 39, 5, .611, 1.04, 11.0, 2.2, 5.0, 30.4, 6.0, 4.59, 6.16],
]

RYU_ROWS = [
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

CHOI_ROWS = [
    [2025, 4.40, 17, 77 + 2/3, 3, 3, 0, 0, 53, 72, 7, 41, 34, 5, .500, 1.36, 6.1, 3.9, 1.6, 15.9, 10.2, 1.05, 0.68],
    [2026, 3.20, 27, 146 + 1/3, 14, 5, 0, 0, 129, 143, 8, 65, 54, 7, .737, 1.35, 7.9, 3.3, 2.4, 20.4, 8.5, 3.52, 3.14],
]

PLAYERS = {
    "구자욱": {
        "유형": "타자", "팀": "삼성 라이온즈", "포지션": "외야수 (우투좌타)", "등번호": 5,
        "시즌기록": pd.DataFrame(GU_ROWS, columns=HIT_COLS),
        "랭킹": {"타율": "1위", "OPS": "3위", "WAR": "6위", "타점": "8위", "득점": "공동 8위", "안타": "8위"},
        "존차트": {
            "HOT & COLD ZONE (타율)": {
                "outer": [.455, .100, .375, .258],
                "inner": [[.258, .310, .342], [.481, .389, .364], [.452, .477, .429]],
                "cmap": "coolwarm", "fmt": "{:.3f}"},
            "삼진 분포도 (%)": {
                "outer": [2.8, 12.7, 15.5, 16.9],
                "inner": [[2.8, 5.6, 15.5], [1.4, 2.8, 9.9], [7.0, 5.6, 1.4]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "타구분포": {"레프트": 36.3, "센터": 28.5, "라이트": 35.2},
        "VS팀": {"한화": .576, "SSG": .457, "두산": .405, "LG": .392, "롯데": .344,
                "KIA": .340, "KT": .333, "NC": .288, "키움": .286},
        "VS팀_라벨": "타율",
        "비고": "",
    },
    "김도영": {
        "유형": "타자", "팀": "KIA 타이거즈", "포지션": "내야수 (우투우타)", "등번호": 5,
        "시즌기록": pd.DataFrame(KIM_ROWS, columns=HIT_COLS),
        "랭킹": {"홈런": "1위", "득점": "2위", "OPS": "2위", "WAR": "3위",
                "타점": "공동 4위", "타율": "9위", "안타": "공동 11위"},
        "존차트": {
            "HOT & COLD ZONE (타율)": {
                "outer": [.296, .083, .171, .059],
                "inner": [[.282, .356, .333], [.333, .500, .333], [.264, .238, .429]],
                "cmap": "coolwarm", "fmt": "{:.3f}"},
            "삼진 분포도 (%)": {
                "outer": [9.7, 9.7, 15.5, 13.6],
                "inner": [[12.6, 4.9, 5.8], [2.9, 1.0, 1.0], [9.7, 7.8, 5.8]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "타구분포": {"레프트": 57.3, "센터": 24.1, "라이트": 18.6},
        "VS팀": {"키움": .448, "KT": .368, "한화": .328, "롯데": .322, "삼성": .306,
                "두산": .255, "SSG": .255, "NC": .244, "LG": .208},
        "VS팀_라벨": "타율",
        "비고": "2025 시즌은 30경기만 출전해서 수치가 작아요.",
    },
    "오스틴 딘": {
        "유형": "타자", "팀": "LG 트윈스", "포지션": "내야수 (우투우타)", "등번호": 23,
        "시즌기록": pd.DataFrame(AUSTIN_ROWS, columns=HIT_COLS),
        "랭킹": {"OPS": "1위", "WAR": "1위", "홈런": "3위", "타점": "3위",
                "안타": "3위", "타율": "4위", "득점": "4위"},
        "존차트": {
            "HOT & COLD ZONE (타율)": {
                "outer": [.308, .219, .164, .364],
                "inner": [[.333, .310, .409], [.364, .302, .300], [.358, .395, .473]],
                "cmap": "coolwarm", "fmt": "{:.3f}"},
            "삼진 분포도 (%)": {
                "outer": [6.3, 6.3, 29.5, 4.2],
                "inner": [[4.2, 4.2, 1.1], [5.3, 5.3, 5.3], [9.5, 7.4, 11.6]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "타구분포": {"레프트": 50.4, "센터": 22.8, "라이트": 26.8},
        "VS팀": {"KT": .403, "키움": .385, "한화": .367, "SSG": .351, "두산": .348,
                "롯데": .339, "NC": .304, "삼성": .246, "KIA": .228},
        "VS팀_라벨": "타율",
        "비고": "KBO 입단 이후(2023~2026) 기록만 있어요.",
    },
    "곽빈": {
        "유형": "투수", "팀": "두산 베어스", "포지션": "투수 (우투우타)", "등번호": 47,
        "시즌기록": pd.DataFrame(KWAK_ROWS, columns=PIT_COLS),
        "랭킹": {"평균자책": "1위", "탈삼진": "1위", "WHIP": "1위", "WAR": "1위",
                "다승": "공동 8위", "이닝": "8위"},
        "존차트": {
            "투구 분포도 (%)": {
                "outer": [9.5, 17.6, 13.9, 8.9],
                "inner": [[5.9, 5.9, 7.8], [5.1, 5.8, 5.5], [4.8, 4.5, 5.0]],
                "cmap": "Oranges", "fmt": "{:.1f}%"},
            "구역별 피안타율": {
                "outer": [.094, .068, .145, .133],
                "inner": [[.278, .264, .115], [.333, .380, .209], [.225, .324, .189]],
                "cmap": "coolwarm_r", "fmt": "{:.3f}"},
            "탈삼진 분포도 (%)": {
                "outer": [7.1, 10.2, 13.7, 11.7],
                "inner": [[6.6, 6.1, 10.7], [3.6, 5.1, 6.1], [7.1, 4.6, 7.6]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "구종": pd.DataFrame(
            {"구종": ["직구", "커터", "슬라이더", "커브", "체인지업"],
             "선수 구속": [152, 146, 136, 121, 134],
             "리그평균 구속": [145, 138, 132, 120, 129],
             "구사비율": [43.2, 17.5, 14.1, 12.8, 12.4]}),
        "VS팀": {"KT": .75, "KIA": .95, "LG": 1.96, "롯데": 2.00, "SSG": 2.08,
                "삼성": 2.25, "한화": 2.30, "키움": 3.00, "NC": 7.20},
        "VS팀_라벨": "평균자책점",
        "비고": "2019·2020 시즌 기록이 표에 없어서 그래프에서 해당 연도가 빠져 있어요.",
    },
    "류현진": {
        "유형": "투수", "팀": "한화 이글스", "포지션": "투수 (좌투우타)", "등번호": 99,
        "시즌기록": pd.DataFrame(RYU_ROWS, columns=PIT_COLS),
        "랭킹": {"다승": "공동 14위", "WAR": "14위", "이닝": "20위"},
        "존차트": {
            "투구 분포도 (%)": {
                "outer": [12.0, 11.4, 9.9, 12.9],
                "inner": [[6.0, 5.4, 7.9], [5.7, 5.7, 5.9], [6.0, 4.2, 6.9]],
                "cmap": "Oranges", "fmt": "{:.1f}%"},
            "구역별 피안타율": {
                "outer": [.111, .217, .088, .143],
                "inner": [[.333, .400, .289], [.268, .356, .348], [.375, .214, .222]],
                "cmap": "coolwarm_r", "fmt": "{:.3f}"},
            "탈삼진 분포도 (%)": {
                "outer": [10.4, 6.1, 20.9, 10.4],
                "inner": [[5.2, 5.2, 7.8], [5.2, 0.0, 4.3], [7.0, 4.3, 13.0]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "구종": pd.DataFrame(
            {"구종": ["직구", "체인지업", "커터", "커브", "스위퍼", "투심"],
             "선수 구속": [142, 127, 135, 111, 123, 139],
             "리그평균 구속": [145, 129, 138, 120, 128, 143],
             "구사비율": [44.0, 24.5, 17.4, 10.8, 2.8, 0.4]}),
        "VS팀": {"SSG": 2.60, "롯데": 2.81, "KIA": 3.00, "KT": 3.42, "NC": 4.38,
                "두산": 4.91, "키움": 6.10, "LG": 18.00},
        "VS팀_라벨": "평균자책점",
        "비고": "KBO 기록만 있어요. 2013~2023년은 표에 없어서 점으로만 표시돼요. 2006년 일부 지표는 원본에서 0으로 표시된 값이에요.",
    },
    "최민석": {
        "유형": "투수", "팀": "두산 베어스", "포지션": "투수 (우투우타)", "등번호": 68,
        "시즌기록": pd.DataFrame(CHOI_ROWS, columns=PIT_COLS),
        "랭킹": {"평균자책": "3위", "다승": "3위", "WAR": "10위",
                "탈삼진": "13위", "WHIP": "13위", "이닝": "15위"},
        "존차트": {
            "투구 분포도 (%)": {
                "outer": [11.0, 13.2, 17.2, 5.1],
                "inner": [[8.1, 6.2, 6.9], [7.3, 5.0, 4.6], [7.3, 4.2, 3.9]],
                "cmap": "Oranges", "fmt": "{:.1f}%"},
            "구역별 피안타율": {
                "outer": [.190, .139, .111, .261],
                "inner": [[.169, .267, .222], [.224, .346, .350], [.125, .359, .564]],
                "cmap": "coolwarm_r", "fmt": "{:.3f}"},
            "탈삼진 분포도 (%)": {
                "outer": [3.9, 17.1, 12.4, 3.9],
                "inner": [[16.3, 1.6, 9.3], [12.4, 3.9, 0.8], [14.7, 2.3, 1.6]],
                "cmap": "Reds", "fmt": "{:.1f}%"},
        },
        "구종": pd.DataFrame(
            {"구종": ["투심", "커터", "스위퍼", "포크", "슬라이더"],
             "선수 구속": [143, 139, 127, 136, 130],
             "리그평균 구속": [143, 138, 128, 132, 132],
             "구사비율": [46.8, 27.3, 11.9, 11.7, 2.2]}),
        "VS팀": {"롯데": 0.00, "한화": 1.64, "KIA": 2.08, "키움": 2.20, "NC": 2.77,
                "KT": 3.71, "SSG": 3.86, "삼성": 5.68, "LG": 5.74},
        "VS팀_라벨": "평균자책점",
        "비고": "시즌이 2개(2025, 2026)뿐이라 추세를 해석하기엔 자료가 적어요.",
    },
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
    pos = [(0, 2), (2, 2), (0, 0), (2, 0)]  # 좌상, 우상, 좌하, 우하
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
    for name, a1, a2, v in [("레프트", 105, 135, left), ("센터", 75, 105, center),
                            ("라이트", 45, 75, right)]:
        ax.add_patch(Wedge((0, 0), 10, a1, a2, facecolor=cm(norm(v)), edgecolor="white", lw=2))
        mid = np.deg2rad((a1 + a2) / 2)
        ax.text(6.5 * np.cos(mid), 6.5 * np.sin(mid), f"{name}\n{v:.1f}%",
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
    ax.set_ylim(100, max(df["선수 구속"]) + 10)
    ax.set_ylabel("km/h")
    ax.set_title(f"{name} 구종별 구속 vs 리그평균")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def pitch_ratio_chart(df, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(df["구사비율"], labels=df["구종"], autopct="%1.1f%%", startangle=90,
           counterclock=False, wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.set_title(f"{name} 구사비율")
    fig.tight_layout()
    return fig


def trend_chart(df, metrics, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    # 연도가 중간에 비면(예: 2012 -> 2024) 선을 잇지 않고 점만 찍는다
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
# 화면
# ===============================================================
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
                st.caption("※ 구역별 피안타율은 낮을수록 투수에게 유리해서, 낮은 칸을 빨간색으로 표시했어요 (네이버 화면과 같은 방식).")

        with tab2:
            a, b = st.columns(2)
            if is_pitcher:
                with a:
                    st.pyplot(pitch_speed_chart(p["구종"], name))
                with b:
                    st.pyplot(pitch_ratio_chart(p["구종"], name))
                st.pyplot(vs_team_chart(p["VS팀"], name, p["VS팀_라벨"]))
                st.caption("평균자책점은 낮을수록 좋아요. 표본이 적은 팀(이닝이 적은 상대)의 값은 믿기 어려워요.")
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
