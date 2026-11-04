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

# ---------------------------------------------------------------
# 실제 데이터 (네이버 스포츠 구자욱 기록 페이지 저장본에서 옮김)
# 다른 선수는 HTML을 받은 뒤 같은 형식으로 추가합니다.
# ---------------------------------------------------------------
SEASON_COLS = ["시즌", "타율", "경기수", "타수", "안타", "2루타", "3루타", "홈런",
               "타점", "득점", "도루", "볼넷", "삼진", "출루율", "장타율", "OPS",
               "IsoP", "BABIP", "wOBA", "wRC+", "WPA", "WAR"]

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

PLAYERS = {
    "구자욱": {
        "팀": "삼성 라이온즈",
        "포지션": "외야수 (우투좌타)",
        "등번호": 5,
        "시즌기록": pd.DataFrame(GU_ROWS, columns=SEASON_COLS),
        "통산": {"타율": .322, "경기수": 1466, "안타": 1820, "홈런": 201, "타점": 984, "득점": 1073, "도루": 150},
        "랭킹_2026": {"타율": "1위", "OPS": "3위", "WAR": "6위", "타점": "8위", "득점": "공동 8위", "안타": "8위"},
        # 순서: 좌상, 우상, 좌하, 우하 / 안쪽 3x3 (위→아래, 왼→오)
        "핫콜드": {"outer": [.455, .100, .375, .258],
                  "inner": [[.258, .310, .342], [.481, .389, .364], [.452, .477, .429]]},
        "삼진분포": {"outer": [2.8, 12.7, 15.5, 16.9],
                    "inner": [[2.8, 5.6, 15.5], [1.4, 2.8, 9.9], [7.0, 5.6, 1.4]]},
        "타구분포": {"레프트": 36.3, "센터": 28.5, "라이트": 35.2},
        "VS팀": {"한화": .576, "SSG": .457, "두산": .405, "LG": .392, "롯데": .344,
                "KIA": .340, "KT": .333, "NC": .288, "키움": .286},
    },
}


# ---------------------------------------------------------------
# 그래프 함수
# ---------------------------------------------------------------
def zone_heatmap(data, title, cmap, fmt):
    outer, inner = data["outer"], data["inner"]
    vals = outer + [v for row in inner for v in row]
    norm = plt.Normalize(min(vals), max(vals))
    cm = plt.get_cmap(cmap)

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


def field_chart(data):
    left, center, right = data["레프트"], data["센터"], data["라이트"]
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


def trend_chart(df, metrics, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in metrics:
        ax.plot(df["시즌"], df[m], marker="o", label=m)
    ax.set_xticks(df["시즌"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def vs_team_chart(data, name):
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([k for k, _ in items], [v for _, v in items], color="#77a4f0")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_title(f"{name} 팀별 타율")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# 화면
# ---------------------------------------------------------------
st.title("⚾ KBO 선수 기록실")
st.caption("선수 이름을 입력하면 등록된 실제 기록을 그래프로 보여줍니다. (데이터 출처: 네이버 스포츠 선수 기록 페이지)")

query = st.text_input("선수 이름", placeholder="예: 구자욱").strip()
st.write("등록된 선수:", ", ".join(PLAYERS.keys()))

if query:
    matches = [n for n in PLAYERS if query in n]
    if not matches:
        st.warning(
            f"'{query}' 선수는 아직 등록되지 않았어요. "
            "그 선수의 기록 페이지를 저장한 HTML을 보내주시면 추가할 수 있어요."
        )
    else:
        name = matches[0] if len(matches) == 1 else st.selectbox("선수 선택", matches)
        p = PLAYERS[name]
        df = p["시즌기록"]

        st.header(f"{name}  NO.{p['등번호']}")
        st.write(f"{p['팀']} | {p['포지션']}")

        latest = df.iloc[-1]
        c = st.columns(5)
        c[0].metric(f"{int(latest['시즌'])} 타율", f"{latest['타율']:.3f}")
        c[1].metric("홈런", int(latest["홈런"]))
        c[2].metric("타점", int(latest["타점"]))
        c[3].metric("OPS", f"{latest['OPS']:.3f}")
        c[4].metric("WAR", f"{latest['WAR']:.2f}")

        st.subheader("🏆 시즌 랭킹")
        st.write(" · ".join(f"**{k}** {v}" for k, v in p["랭킹_2026"].items()))

        tab1, tab2, tab3, tab4 = st.tabs(["🔥 존 히트맵", "⚾ 타구·팀별", "📈 시즌 추이", "📄 표/CSV"])

        with tab1:
            a, b = st.columns(2)
            with a:
                st.pyplot(zone_heatmap(p["핫콜드"], f"{name} HOT & COLD ZONE (타율)",
                                       "coolwarm", "{:.3f}"))
            with b:
                st.pyplot(zone_heatmap(p["삼진분포"], f"{name} 삼진 분포도 (%)",
                                       "Reds", "{:.1f}%"))

        with tab2:
            a, b = st.columns(2)
            with a:
                st.pyplot(field_chart(p["타구분포"]))
            with b:
                st.pyplot(vs_team_chart(p["VS팀"], name))

        with tab3:
            cols = [c_ for c_ in df.columns if c_ != "시즌"]
            chosen = st.multiselect("항목 선택", cols, default=["타율", "출루율", "장타율"])
            if chosen:
                st.pyplot(trend_chart(df, chosen, f"{name} 시즌별 기록"))
            st.caption("※ 2015·2016 시즌은 원본에서 wRC+와 WAR이 0으로 표시돼 있어요(자료 없음으로 추정).")

        with tab4:
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ 시즌 기록 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{name}_시즌기록.csv", mime="text/csv")
