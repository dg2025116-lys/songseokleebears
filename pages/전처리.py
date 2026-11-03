import io
import subprocess
import sys
import zipfile

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="네이버 선수 기록 CSV 변환기", page_icon="⚾")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------
# 1) HTML 파싱
# ---------------------------------------------------------------
def text(el):
    return el.get_text(strip=True) if el else ""


def get_player_name(soup):
    el = soup.select_one('[class*="PlayerEndHeader_player__"]')
    return text(el) or "player"


def parse_tables(soup, prefix=""):
    """<table> 태그를 모두 DataFrame으로 변환"""
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
    return name, data


# ---------------------------------------------------------------
# 2) Playwright로 페이지 열기
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="브라우저 준비 중... (최초 1회)")
def install_chromium():
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
    )
    return True


def render_page(url, include_game_log=False):
    """렌더링된 HTML 문자열 리스트를 반환 (통산기록, [경기별 기록])"""
    from playwright.sync_api import sync_playwright

    install_chromium()
    htmls = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=40000)
        page.wait_for_selector("table", timeout=20000)
        htmls.append(("", page.content()))

        if include_game_log:
            try:
                page.get_by_role("radio", name="경기별 기록").click()
                page.wait_for_timeout(2000)
                htmls.append(("경기별_", page.content()))
            except Exception:
                st.warning("경기별 기록 탭을 클릭하지 못했어요.")

        browser.close()
    return htmls


# ---------------------------------------------------------------
# 3) CSV / ZIP 변환
# ---------------------------------------------------------------
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")  # 엑셀 한글 깨짐 방지


def make_zip(tables):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            safe = name.replace("/", "_").replace(" ", "_")
            z.writestr(f"{safe}.csv", to_csv_bytes(df))
    return buf.getvalue()


# ---------------------------------------------------------------
# 4) 화면
# ---------------------------------------------------------------
st.title("⚾ 네이버 선수 기록 CSV 변환기")
st.caption("학습용 프로젝트입니다. 이용약관을 지키고 과도한 요청은 피하세요.")

mode = st.radio(
    "입력 방식",
    ["링크 입력 (브라우저로 자동 접속)", "저장한 HTML 파일 업로드 (가장 안정적)"],
)

if mode.startswith("링크"):
    url = st.text_input(
        "선수 기록 페이지 링크",
        placeholder="https://m.sports.naver.com/player/index?playerId=62404&category=kbo&tab=record",
    )
    include_game_log = st.checkbox("경기별 기록도 가져오기 (시간이 더 걸려요)")

    if st.button("CSV 만들기", type="primary") and url:
        if "naver.com" not in url:
            st.error("네이버 스포츠 링크를 입력해 주세요.")
        else:
            try:
                with st.spinner("페이지를 여는 중... 최대 1분 정도 걸릴 수 있어요"):
                    htmls = render_page(url.strip(), include_game_log)
                tables, player = {}, "player"
                for prefix, html in htmls:
                    player, part = parse_html(html, prefix)
                    tables.update(part)
                st.session_state["tables"] = tables
                st.session_state["player"] = player
            except Exception as e:
                st.error(f"실패: {e}")
                st.info("링크 방식이 안 되면 HTML 파일 업로드 방식을 사용해 보세요.")

else:
    up = st.file_uploader("브라우저에서 '다른 이름으로 저장'한 .html 파일", type=["html", "htm"])
    if up is not None:
        html = up.read().decode("utf-8", errors="ignore")
        player, tables = parse_html(html)
        st.session_state["tables"] = tables
        st.session_state["player"] = player

# ---------------------------------------------------------------
# 5) 결과
# ---------------------------------------------------------------
if st.session_state.get("tables"):
    tables = st.session_state["tables"]
    player = st.session_state["player"]

    st.success(f"**{player}** 선수 데이터에서 {len(tables)}개 표를 찾았어요.")

    st.download_button(
        "⬇️ 전체 CSV 묶음(ZIP) 다운로드",
        data=make_zip(tables),
        file_name=f"{player}_records.zip",
        mime="application/zip",
    )

    st.divider()
    for name, df in tables.items():
        st.subheader(name)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"⬇️ {name}.csv",
            data=to_csv_bytes(df),
            file_name=f"{player}_{name}.csv".replace(" ", "_"),
            mime="text/csv",
            key=f"dl_{name}",
        )
elif "tables" in st.session_state:
    st.warning("표를 찾지 못했어요. 링크나 파일을 확인해 주세요.")
