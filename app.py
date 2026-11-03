"""Streamlit 앱: KBO 대표 선수 히트맵 · 잠실야구장 에디션

실행:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import streamlit as st  # noqa: E402

from kbo_heatmap import classify as K  # noqa: E402
from kbo_heatmap import config as C  # noqa: E402
from kbo_heatmap import plots, render  # noqa: E402
from kbo_heatmap.data import load_pitches_bytes, load_players, make_demo  # noqa: E402
from kbo_heatmap.tables import summary_tables  # noqa: E402

st.set_page_config(page_title="KBO 대표 선수 히트맵 · 잠실야구장", page_icon="⚾", layout="wide")

TEMPLATE_PATH = C.REPO_ROOT / "data" / "pitches_template.csv"


# ============================================================ 캐시된 계산
@st.cache_resource
def get_teams():
    return load_players()


@st.cache_resource
def init_font():
    return plots.setup_font(os.environ.get("KBO_FONT_PATH") or None)


@st.cache_data(show_spinner="데모 데이터를 만드는 중…")
def demo_df(seed: int = 2026):
    return make_demo(get_teams(), seed=seed)


@st.cache_data(show_spinner="CSV를 검증하는 중…")
def parse_upload(raw: bytes):
    return load_pitches_bytes(raw)


@st.cache_data(show_spinner=False)
def hero(demo: bool) -> bytes:
    return render.hero_png(demo=demo)


# 앞에 _가 붙은 인자(_df)는 캐시 키 계산에서 제외됩니다. 대신 data_key로 데이터를 구분합니다.
@st.cache_data(show_spinner="히트맵을 그리는 중…", max_entries=256)
def batter_view(data_key: str, _df, name: str, team_name: str, grid: int, iso: float, demo: bool):
    return render.batter_png(_df, name, team_name, grid, iso, demo)


@st.cache_data(show_spinner="히트맵을 그리는 중…", max_entries=256)
def pitcher_view(data_key: str, _df, name: str, team_name: str, grid: int, demo: bool):
    return render.pitcher_png(_df, name, team_name, grid, demo)


@st.cache_data(show_spinner="분류 모델을 학습하는 중… (몇 초 걸려요)", max_entries=8)
def classification(data_key: str, _df):
    # 웹앱에서는 속도를 위해 교차검증을 생략하고 나무 수/표본 수를 줄입니다.
    return K.train_and_evaluate(_df, cv=0, rf_trees=100, max_rows=60_000)


@st.cache_data(show_spinner=False, max_entries=8)
def classification_outputs(data_key: str, _result, demo: bool):
    return (render.evaluation_png(_result, demo), render.probability_map_png(_result, demo),
            K.build_report(_result, demo))


@st.cache_data(show_spinner=False, max_entries=64)
def tables(data_key: str, _df, grid: int, iso: float):
    return summary_tables(_df, get_teams(), grid, iso)


# ============================================================ 사이드바
def sidebar():
    """(데이터프레임, data_key, demo 여부, 선택한 설정들)"""
    sb = st.sidebar
    sb.header("⚾ 설정")

    source = sb.radio("데이터", ["DEMO (가짜 데이터)", "내 CSV 업로드"], key="source")
    df, key, demo = None, "demo", True
    if source.startswith("내"):
        up = sb.file_uploader("투구 데이터 CSV", type=["csv"], key="upload")
        if up is None:
            sb.info("CSV를 올리기 전까지는 DEMO 데이터를 보여드려요.")
        else:
            try:
                raw = up.getvalue()
                df = parse_upload(raw)
                key, demo = "up-" + hashlib.md5(raw).hexdigest(), False
                sb.success(f"불러옴: {len(df):,}구")
            except Exception as e:  # 형식 오류는 앱을 멈추지 않고 안내만 한다
                sb.error(f"CSV를 읽지 못했어요: {e}")
                sb.caption("형식은 '사용법' 탭에서 확인하세요. 그동안 DEMO 데이터를 보여드려요.")
    if df is None:
        df, key, demo = demo_df(), "demo", True

    grid = sb.radio("히트맵 분할", list(C.SUPPORTED_GRIDS), index=list(C.SUPPORTED_GRIDS).index(C.DEFAULT_GRID),
                    format_func=lambda g: f"{g}×{g}", key="grid", horizontal=True)
    iso = round(sb.slider("장타형 기준 (ISO ≥)", 0.05, 0.30, float(C.ISO_THRESHOLD), 0.005, key="iso"), 3)

    sb.divider()
    teams = get_teams()
    by_code = {t.code: t for t in teams}
    code = sb.selectbox("구단", list(by_code), format_func=lambda c: by_code[c].name, key="team")
    role = sb.radio("선수 구분", ["타자", "투수"], key="role", horizontal=True)

    font = init_font()
    if not font:
        sb.warning("한글 폰트를 찾지 못했어요. 그림의 한글이 깨질 수 있어요. "
                   "저장소의 packages.txt(fonts-nanum)를 확인하세요.")
    return df, key, demo, grid, iso, by_code[code], role


# ============================================================ 본문
def _label(player, has_data: bool) -> str:
    tag = "" if has_data else "  (데이터 없음)"
    return f"{player.name}{'' if player.verified else ' ⚠'}{tag}"


def show_batter(df, key, demo, grid, iso, team, player):
    png, s = batter_view(key, df, player.name, team.name, grid, iso, demo)
    if s is None:
        st.info(f"'{player.name}'의 타석 결과(pa_result)가 데이터에 없어 타자 분석을 할 수 없어요.")
        return
    cols = st.columns(5)
    cols[0].metric("타율", f"{s['avg']:.3f}")
    cols[1].metric("장타율", f"{s['slg']:.3f}")
    cols[2].metric("ISO", f"{s['iso']:.3f}")
    cols[3].metric("유형", s["type"])
    cols[4].metric("타수", f"{s['ab']}")
    st.markdown(f"**강한 존** {s['strong_zone']}　·　**약한 존** {s['weak_zone']}　_(포수 시점)_")
    if not s["sample_ok"]:
        st.warning(f"타수가 {s['ab']}로 적어서(기준 {C.MIN_AB_CLASSIFY}) 유형 판정은 참고용이에요.")
    if player.ref:
        st.caption(f"참고 시즌 기록: 타율 {player.ref.get('avg', '-')} · OPS {player.ref.get('ops', '-')} "
                   "(공개 순위표, 위 계산과는 별개)")
    st.image(png)


def show_pitcher(df, key, demo, grid, team, player):
    png, s = pitcher_view(key, df, player.name, team.name, grid, demo)
    if s is None:
        st.info(f"'{player.name}'의 투구가 데이터에 없어요.")
        return
    cols = st.columns(4)
    cols[0].metric("투구 수", f"{s['pitches']:,}")
    cols[1].metric("직구 비율", f"{s['fb_share']:.0%}")
    cols[2].metric("직구 스트라이크율", f"{s['fb_strike']:.0%}")
    cols[3].metric("변화구 스트라이크율", f"{s['br_strike']:.0%}")
    st.markdown(f"**직구 잘 넣는 존** {s['fb_best_zone']}　·　**변화구 잘 넣는 존** {s['br_best_zone']}　_(포수 시점)_")
    if player.ref.get("era") is not None:
        st.caption(f"참고 시즌 기록: 평균자책점 {player.ref['era']:.2f} (공개 순위표, 위 계산과는 별개)")
    st.image(png)


def tab_players(df, key, demo, grid, iso, team, role):
    players = team.batters if role == "타자" else team.pitchers
    if not players:
        st.info(f"{team.name}은 설정 파일(data/players.json)에 대표 {role}가 없어요.")
        return
    names_with_data = set(df["batter" if role == "타자" else "pitcher"].unique())
    by_name = {p.name: p for p in players}
    name = st.selectbox(f"{role} 선택", list(by_name), key=f"player_{team.code}_{role}",
                        format_func=lambda n: _label(by_name[n], n in names_with_data))
    player = by_name[name]
    if not player.verified:
        st.caption(f"⚠ {player.note or '2026 기록을 확인하지 못한 선수예요.'}")
    try:
        if role == "타자":
            show_batter(df, key, demo, grid, iso, team, player)
        else:
            show_pitcher(df, key, demo, grid, team, player)
    except Exception as e:  # 그림 하나가 실패해도 앱 전체가 멈추지 않게
        st.error(f"이 선수의 그림을 만들지 못했어요: {e}")
        with st.expander("자세한 오류"):
            st.exception(e)


def tab_summary(df, key, grid, iso):
    bt, pt = tables(key, df, grid, iso)
    st.subheader("타자")
    if bt.empty:
        st.info("표시할 타자 데이터가 없어요.")
    else:
        st.dataframe(bt, hide_index=True)
        st.download_button("타자 요약 CSV 내려받기", bt.to_csv(index=False).encode("utf-8-sig"),
                           file_name="batter_summary.csv", mime="text/csv", key="dl_b")
    st.subheader("투수")
    if pt.empty:
        st.info("표시할 투수 데이터가 없어요.")
    else:
        st.dataframe(pt, hide_index=True)
        st.download_button("투수 요약 CSV 내려받기", pt.to_csv(index=False).encode("utf-8-sig"),
                           file_name="pitcher_summary.csv", mime="text/csv", key="dl_p")


def tab_classify(df, key, demo):
    st.markdown(
        "투구가 홈플레이트를 지날 때의 **위치와 구종**만 보고 **스트라이크 / 볼**을 맞히는 **분류 모델**을 학습해요. "
        "정답(라벨)은 `is_strike`이고, 학습에 쓰지 않은 **테스트 데이터**로만 성능을 평가해요.")
    if st.button("분류 모델 학습하기", key="train"):
        st.session_state["cls_ready"] = key
    if st.session_state.get("cls_ready") != key:
        st.info("버튼을 누르면 5개 모델(기준선, 로지스틱 회귀 2종, 결정트리, 랜덤포레스트)을 학습하고 비교해요.")
        return
    try:
        result = classification(key, df)
        eval_png, map_png, report = classification_outputs(key, result, demo)
    except ValueError as e:      # 데이터가 너무 적거나 정답이 한 가지뿐인 경우 등
        st.error(f"분류를 할 수 없어요: {e}")
        return
    except Exception as e:
        st.error(f"분류 중 오류가 났어요: {e}")
        with st.expander("자세한 오류"):
            st.exception(e)
        return
    b = result.metrics.loc[result.best_name]
    cols = st.columns(4)
    cols[0].metric("최고 모델", result.best_name)
    cols[1].metric("정확도", f"{b['정확도']:.3f}", f"기준선 대비 {b['정확도'] - result.baseline_accuracy:+.3f}")
    cols[2].metric("ROC-AUC", f"{b['ROC-AUC']:.3f}")
    cols[3].metric("테스트 투구 수", f"{result.n_test:,}")
    if b["정확도"] <= result.baseline_accuracy:
        st.warning("최고 모델이 기준선(항상 다수 범주를 찍는 모델)보다 낫지 않아요. 의미 있게 학습하지 못한 거예요.")
    st.subheader("모델 비교 (테스트 데이터)")
    st.dataframe(result.metrics.round(3))
    st.image(eval_png)
    st.image(map_png)
    st.download_button("분류 결과 보고서(.md) 내려받기", report.encode("utf-8"),
                       file_name="classification_report.md", mime="text/markdown", key="dl_report")


def tab_help():
    st.markdown(
        """
**데이터 형식 (투구 1개 = 1행)**

| 컬럼 | 설명 |
|---|---|
| `pitcher`, `batter` | 투수·타자 이름 (`data/players.json`과 글자까지 같아야 함) |
| `pitch_type` | `직구` / `변화구` (FF, SL, 슬라이더, 커브 등도 자동 분류) |
| `plate_x`, `plate_z` | 홈플레이트 통과 위치(ft). 포수 시점, 오른쪽이 +. 없으면 `zone`(A1~E5) 사용 |
| `is_strike` | 스트라이크 1, 볼 0 (루킹·헛스윙·파울 포함) |
| `pa_result` | 그 투구로 타석이 끝났을 때만 `1루타/2루타/3루타/홈런/아웃/삼진/볼넷…`, 아니면 빈칸 |

CSV는 UTF-8과 CP949(엑셀 한글 저장)를 모두 읽어요. 구종을 알 수 없거나 위치가 빈 행은 자동으로 건너뜁니다.
"""
    )
    if TEMPLATE_PATH.exists():
        st.download_button("CSV 예시 파일 받기", TEMPLATE_PATH.read_bytes(),
                           file_name="pitches_template.csv", mime="text/csv", key="dl_t")
    st.markdown(
        f"""
**지표 기준**
- 장타형: ISO(장타율−타율) ≥ 사이드바 값 (기본 {C.ISO_THRESHOLD}, 임의 기준이니 조정하세요)
- 강한/약한 존: 칸당 타수 {C.MIN_AB_BEST} 이상인 칸 중 최고/최저 타율
- 잘 넣는 존: 칸당 {C.MIN_PITCH_BEST}구 이상인 칸 중 스트라이크율 최고
- 좌표는 포수 시점이며 좌/우타자는 구분하지 않아요.
"""
    )


def main():
    df, key, demo, grid, iso, team, role = sidebar()
    st.image(hero(demo))
    if demo:
        st.warning("**DEMO DATA** — 지금 보이는 모든 수치와 히트맵은 가짜 데이터로 만든 예시예요. "
                   "실제 선수의 기록이 아닙니다. 왼쪽에서 '내 CSV 업로드'를 선택해 실제 투구 데이터를 올리세요.")
    t1, t2, t3, t4 = st.tabs(["선수 히트맵", "전체 요약표", "분류 모델 (스트라이크/볼)", "사용법·데이터 형식"])
    with t1:
        tab_players(df, key, demo, grid, iso, team, role)
    with t2:
        tab_summary(df, key, grid, iso)
    with t3:
        tab_classify(df, key, demo)
    with t4:
        tab_help()
    st.caption("KBO와 무관한 비공식 학습용 프로젝트 · 잠실야구장 배경은 필드 구조를 형상화한 일러스트입니다.")


main()
