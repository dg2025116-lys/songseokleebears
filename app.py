import io
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from matplotlib.patches import Rectangle, Wedge

st.set_page_config(page_title="네이버 선수 기록 분석기", page_icon="⚾", layout="wide")


# ---------------------------------------------------------------
# 한글 폰트 (캐시해서 한 번만 실행)
# ---------------------------------------------------------------
@st.cache_resource
def setup_korean_font():
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return True


setup_korean_font()


# ---------------------------------------------------------------
# 1) HTML 파싱
# ---------------------------------------------------------------
def text(el):
    return el.get_text(strip=True) if el else ""


def get_player_name(soup):
    el = soup.select_one('[class*="PlayerEndHeader_player__"]')
    return text(el) or "player"


def parse_tables(soup, prefix=""):
    result = {}
    for i, table in enumerate(soup.find_all("table"), start=1):
        header = [text(th) for th in table.select("thead th")]
        rows = []
        for tr in table.select("tbody tr"):
            cells = [text(c) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        if len(header) == len(rows[0]):
            df = pd.DataFrame(rows, columns=header)
        else:
            df = pd.DataFrame(rows)
        title_tag = table.find_previous(["h3", "h4"])
        title = text(title_tag) or "표"
        result[f"{prefix}{i}_{title}"] = df
    return result


def parse_vs_team(soup):
    rows = []
    for li in soup.select('li[class*="BarChart_item"]'):
        team = text(li.select_one('[class*="team_name"]'))
        value = text(li.select_one('[class*="number"]'))
        if team:
            rows.append({"팀": team, "값": value})
    return pd.DataFrame(rows)


def parse_ranking_summary(soup):
    rows = []
    for li in soup.select('li[class*="RankingSummary_item"]'):
        rows.append(
            {
                "항목": text(li.select_one('[class*="RankingSummary_title"]')),
                "순위": text(li.select_one('[class*="RankingSummary_ranking"]')),
            }
        )
    return pd.DataFrame(rows)


def parse_season_tabs(soup):
    rows = []
    for item in soup.select('[class*="RecordTab_tab_item"]'):
        rows.append(
            {
                "항목": text(item.select_one('[class*="RecordTab_title"]')),
                "기록": text(item.select_one('[class*="RecordTab_record"]')),
            }
        )
    return pd.DataFrame(rows)


OUTER_NAMES = ["바깥_좌상", "바깥_우상", "바깥_좌하", "바깥_우하"]
INNER_NAMES = [f"안쪽_{r}행{c}열" for r in (1, 2, 3) for c in (1, 2, 3)]


def get_step(el):
    for c in el.get("class", []):
        if "SquareChart_step" in c:
            digits = "".join(ch for ch in c.split("SquareChart_step")[1] if ch.isdigit())
            return int(digits) if digits else None
    return None


def parse_square_chart(chart):
    rows = []
    outers = chart.select('[class*="SquareChart_outer"]')
    inners = chart.select('[class*="SquareChart_inner__"]')
    for names, els in ((OUTER_NAMES, outers), (INNER_NAMES, inners)):
        for name, el in zip(names, els):
            rows.append(
                {
                    "구역": name,
                    "값": text(el.select_one('[class*="SquareChart_average"]')),
                    "색상단계": get_step(el),
                }
            )
    return pd.DataFrame(rows)


def parse_charts(soup, prefix=""):
    result = {}
    for group in soup.select('[class*="StatusChart_chart_group"]'):
        title = text(group.select_one('[class*="StatusChart_sub_title"]'))
        if not title:
            continue
        if group.select_one('[class*="SquareChart_chart_box"]'):
            df = parse_square_chart(group)
            if not df.empty:
                result[f"{prefix}차트_{title}"] = df
        elif group.select_one('[class*="GroundChart_chart_area"]'):
            rows = []
            for label, key in [
                ("레프트", "position_lf"),
                ("센터", "position_cf"),
                ("라이트", "position_rf"),
            ]:
                el = group.select_one(f'[class*="GroundChart_{key}"]')
                if el:
                    for b in el.select(".blind"):
                        b.extract()
                    rows.append({"방향": label, "비율": text(el)})
            if rows:
                result[f"{prefix}차트_{title}"] = pd.DataFrame(rows)
    return result


def parse_legends(soup, prefix=""):
    result = {}
    seen = set()
    n = 0
    for item in soup.select('[class*="DonutChartLegend_legend_item"]'):
        parent = item.parent
        if id(parent) in seen:
            continue
        seen.add(id(parent))
        rows = []
        for it in parent.select('[class*="DonutChartLegend_legend_item"]'):
            rows.append(
                {
                    "항목": text(it.select_one('[class*="DonutChartLegend_label"]')),
                    "비율": text(it.select_one('[class*="DonutChartLegend_percentage"]')),
                    "값": text(it.select_one('[class*="DonutChartLegend_value"]')),
                }
            )
        rows = [r for r in rows if r["항목"]]
        if rows:
            n += 1
            title_tag = parent.find_previous(["h3", "h4"])
            title = text(title_tag) or f"도넛{n}"
            result[f"{prefix}비율_{n}_{title}"] = pd.DataFrame(rows)
    return result


def parse_html(html, prefix=""):
    soup = BeautifulSoup(html, "html.parser")
    name = get_player_name(soup)
    data = parse_tables(soup, prefix)
    extras = {
        f"{prefix}시즌_요약": parse_season_tabs(soup),
        f"{prefix}주요_랭킹": parse_ranking_summary(soup),
        f"{prefix}VS팀": parse_vs_team(soup),
    }
    for key, df in extras.items():
        if not df.empty:
            data[key] = df
    data.update(parse_charts(soup, prefix))
    data.update(parse_legends(soup, prefix))
    return name, data


# ---------------------------------------------------------------
# 2) CSV / ZIP
# ---------------------------------------------------------------
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_name(s):
    return s.replace("/", "_").replace(" ", "_").replace("&", "and")


def make_zip(players):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pname, tables in players.items():
            for tname, df in tables.items():
                z.writestr(f"{safe_name(pname)}/{safe_name(tname)}.csv", to_csv_bytes(df))
    return buf.getvalue()


# ---------------------------------------------------------------
# 3) 그래프 함수
# ---------------------------------------------------------------
def to_number(s):
    s = str(s).replace("%", "").replace(",", "").strip()
    m = re.fullmatch(r"(\d+)\s+(\d)/(\d)", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    try:
        return float(s)
    except ValueError:
        return np.nan


def draw_zone_heatmap(df, title, cmap="coolwarm", fmt="{:.3f}"):
    vals = {r["구역"]: to_number(r["값"]) for _, r in df.iterrows()}
    all_vals = [v for v in vals.values() if not np.isnan(v)]
    if not all_vals:
        return None
    norm = plt.Normalize(min(all_vals), max(all_vals))
    colormap = plt.get_cmap(cmap)

    def color(v):
        return colormap(norm(v)) if not np.isnan(v) else "lightgray"

    def label(v):
        return fmt.format(v) if not np.isnan(v) else "-"

    fig, ax = plt.subplots(figsize=(5, 5))
    outer_pos = {"바깥_좌상": (0, 2), "바깥_우상": (2, 2), "바깥_좌하": (0, 0), "바깥_우하": (2, 0)}
    for name, (x, y) in outer_pos.items():
        v = vals.get(name, np.nan)
        ax.add_patch(Rectangle((x, y), 2, 2, facecolor=color(v), edgecolor="white", lw=2))
        tx = x + 0.3 if x == 0 else x + 1.7
        ty = y + 1.7 if y == 2 else y + 0.3
        ax.text(tx, ty, label(v), ha="left" if x == 0 else "right",
                va="center", fontsize=11, fontweight="bold")

    cell = 2 / 3
    start = 2 - 3 * cell / 2
    for i in range(3):
        for j in range(3):
            v = vals.get(f"안쪽_{i + 1}행{j + 1}열", np.nan)
            x = start + j * cell
            y = start + (2 - i) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color(v),
                                   edgecolor="black", lw=1))
            ax.text(x + cell / 2, y + cell / 2, label(v), ha="center",
                    va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "\n(투수 시점)", fontsize=13)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


def draw_field_chart(df):
    vals = {r["방향"]: to_number(r["비율"]) for _, r in df.iterrows()}
    left, center, right = (vals.get(k, 0) for k in ("레프트", "센터", "라이트"))
    top = max(left, center, right)
    norm = plt.Normalize(0, top * 1.2 if top > 0 else 1)
    cmap = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, a1, a2, v in [("레프트", 105, 135, left), ("센터", 75, 105, center),
                            ("라이트", 45, 75, right)]:
        ax.add_patch(Wedge((0, 0), 10, a1, a2, facecolor=cmap(norm(v)),
                           edgecolor="white", lw=2))
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


def draw_donut(df, title):
    d = df.copy()
    pct = d["비율"].apply(to_number)
    val = d["값"].apply(to_number)
    sizes = pct if pct.notna().any() and pct.sum() > 0 else val
    mask = sizes.notna() & (sizes > 0)
    if not mask.any():
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        sizes[mask],
        labels=d["항목"][mask],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        textprops=dict(fontsize=10),
    )
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig


def season_df(tables):
    for name, df in tables.items():
        if name.startswith("경기별_"):
            continue
        if "시즌" in df.columns and len(df.columns) > 3:
            d = df[df["시즌"].astype(str).str.isdigit()].copy()
            if d.empty:
                continue
            d["시즌"] = d["시즌"].astype(int)
            return d.sort_values("시즌")
    return None


def default_metrics(cols):
    for group in (["타율", "출루율", "장타율"], ["평균자책점", "WHIP"], ["평균자책점"]):
        picked = [c for c in group if c in cols]
        if picked:
            return picked
    return cols[:2]


def draw_season_trend(d, metrics, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in metrics:
        ax.plot(d["시즌"], d[m].apply(to_number), marker="o", label=m)
    ax.set_xticks(d["시즌"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def draw_compare(players, metric):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pname, tables in players.items():
        d = season_df(tables)
        if d is not None and metric in d.columns:
            ax.plot(d["시즌"], d[metric].apply(to_number), marker="o", label=pname)
    ax.set_xlabel("시즌")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"선수별 {metric} 비교")
    fig.tight_layout()
    return fig


def draw_vs_team(df, title):
    d = df.copy()
    d["값"] = d["값"].apply(to_number)
    d = d.dropna(subset=["값"]).sort_values("값", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(d["팀"], d["값"], color="#77a4f0")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def find_table(tables, keyword):
    for name, df in tables.items():
        if keyword in name and not name.startswith("경기별_"):
            return df
    return None


# ---------------------------------------------------------------
# 4) 입력 화면 (HTML 업로드 전용)
# ---------------------------------------------------------------
if "players" not in st.session_state:
    st.session_state["players"] = {}


def add_player(name, tables):
    players = st.session_state["players"]
    key, n = name, 2
    while key in players:
        key = f"{name}({n})"
        n += 1
    players[key] = tables


st.title("⚾ 네이버 선수 기록 분석기 (타자 · 투수)")
st.caption("학습용 프로젝트입니다. 네이버 스포츠 선수 기록 페이지를 '다른 이름으로 저장'한 HTML을 올려주세요.")

with st.expander("HTML 저장 방법"):
    st.markdown(
        "1. 크롬으로 선수 기록 페이지를 열고 **끝까지 스크롤**해서 모든 내용이 보이게 합니다.\n"
        "2. `Ctrl+S` → 형식 **'웹페이지, 전체'** 로 저장합니다.\n"
        "3. 저장된 `.html` 파일을 아래에 올립니다. (여러 선수 동시 선택 가능)"
    )

ups = st.file_uploader(
    "저장한 .html 파일 (여러 개 선택 가능)",
    type=["html", "htm"],
    accept_multiple_files=True,
)
if ups and st.button("분석하기", type="primary"):
    for up in ups:
        html = up.read().decode("utf-8", errors="ignore")
        pname, tables = parse_html(html)
        add_player(pname, tables)
        st.success(f"{pname} 선수 추가 완료 ({len(tables)}개 항목)")

if st.session_state["players"]:
    if st.button("선수 목록 모두 지우기"):
        st.session_state["players"] = {}
        st.rerun()


# ---------------------------------------------------------------
# 5) 결과 화면
# ---------------------------------------------------------------
players = st.session_state["players"]

if players:
    st.info("분석된 선수: " + ", ".join(players.keys()))

    tab_one, tab_cmp, tab_table = st.tabs(["📈 선수별 그래프", "🆚 선수 비교", "📄 표 / CSV"])

    with tab_one:
        pname = st.selectbox("선수 선택", list(players.keys()))
        tables = players[pname]
        shown = False

        hot = find_table(tables, "HOT")
        ks = find_table(tables, "삼진")
        if hot is not None or ks is not None:
            c1, c2 = st.columns(2)
            with c1:
                if hot is not None:
                    fig = draw_zone_heatmap(hot, f"{pname} HOT & COLD ZONE (타율)",
                                            cmap="coolwarm", fmt="{:.3f}")
                    if fig:
                        st.pyplot(fig)
                        shown = True
            with c2:
                if ks is not None:
                    fig = draw_zone_heatmap(ks, f"{pname} 삼진 분포도 (%)",
                                            cmap="Reds", fmt="{:.1f}%")
                    if fig:
                        st.pyplot(fig)
                        shown = True

        ground = find_table(tables, "타구")
        if ground is not None:
            st.pyplot(draw_field_chart(ground))
            shown = True

        donut_items = [(n, df) for n, df in tables.items() if "비율_" in n]
        if donut_items:
            st.subheader("비율 차트 (구종 등)")
            cols = st.columns(2)
            for i, (n, df) in enumerate(donut_items):
                title = n.split("_", 2)[-1]
                fig = draw_donut(df, f"{pname} {title}")
                if fig:
                    with cols[i % 2]:
                        st.pyplot(fig)
                        shown = True

        vs = find_table(tables, "VS팀")
        if vs is not None and not vs.empty:
            st.pyplot(draw_vs_team(vs, f"{pname} 팀별 기록"))
            shown = True

        d = season_df(tables)
        if d is not None:
            st.subheader("시즌별 기록 추이")
            cols_ = [c for c in d.columns if c != "시즌"]
            chosen = st.multiselect("항목 선택", cols_,
                                    default=default_metrics(cols_), key=f"trend_{pname}")
            if chosen:
                st.pyplot(draw_season_trend(d, chosen, f"{pname} 시즌별 기록"))
                shown = True

        if not shown:
            st.warning("그래프로 그릴 데이터를 찾지 못했어요. '표 / CSV' 탭을 확인해 보세요.")

    with tab_cmp:
        if len(players) < 2:
            st.write("선수를 2명 이상 추가하면 비교할 수 있어요.")
        else:
            common = None
            for t in players.values():
                d = season_df(t)
                if d is not None:
                    cols_ = set(c for c in d.columns if c != "시즌")
                    common = cols_ if common is None else common & cols_
            if common:
                pref = ["타율", "OPS", "홈런", "타점", "평균자책점", "WHIP", "탈삼진", "WAR"]
                order = [c for c in pref if c in common] + sorted(common - set(pref))
                metric = st.selectbox("비교할 항목", order)
                st.pyplot(draw_compare(players, metric))
            else:
                st.warning("공통 항목이 없어요. 타자끼리, 투수끼리 비교해 보세요.")

    with tab_table:
        st.download_button(
            "⬇️ 전체 선수 CSV 묶음(ZIP) 다운로드",
            data=make_zip(players),
            file_name="players_records.zip",
            mime="application/zip",
        )
        st.divider()
        for pname, tables in players.items():
            with st.expander(f"{pname} 표 보기", expanded=False):
                for name, df in tables.items():
                    st.subheader(name)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        f"⬇️ {name}.csv",
                        data=to_csv_bytes(df),
                        file_name=f"{safe_name(pname)}_{safe_name(name)}.csv",
                        mime="text/csv",
                        key=f"dl_{pname}_{name}",
                    )
