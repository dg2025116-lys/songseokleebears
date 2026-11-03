"""Streamlit 앱을 브라우저 없이 실행해 오류가 없는지 확인합니다(streamlit.testing)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def test_app_starts_with_demo_and_no_exception():
    at = _run()
    assert not at.exception
    assert any("DEMO DATA" in w.value for w in at.warning)
    assert len(at.metric) == 5                      # 첫 화면: KIA 타자 → 지표 5개
    assert at.selectbox(key="team").value == "KIA"


def test_switch_role_team_and_grid():
    at = _run()
    at.radio(key="role").set_value("투수").run()
    assert not at.exception and len(at.metric) == 4
    at.selectbox(key="team").set_value("KIWOOM").run()
    assert not at.exception                          # 키움 투수는 있음
    at.radio(key="role").set_value("타자").run()
    assert not at.exception                          # 키움 타자는 없음 → 안내만 표시
    assert any("대표 타자가 없어요" in i.value for i in at.info)
    at.radio(key="grid").set_value(3).run()
    assert not at.exception


def test_every_player_renders_without_error():
    from kbo_heatmap.data import load_players
    at = _run()
    for t in load_players():
        at.selectbox(key="team").set_value(t.code).run()
        for role, plist in (("타자", t.batters), ("투수", t.pitchers)):
            at.radio(key="role").set_value(role).run()
            assert not at.exception, (t.code, role)
            for p in plist:
                at.selectbox(key=f"player_{t.code}_{role}").set_value(p.name).run()
                assert not at.exception, (t.code, role, p.name)
                assert len(at.error) == 0, (t.code, role, p.name)


def test_iso_slider_changes_type():
    at = _run()
    at.selectbox(key="player_KIA_타자").set_value("김도영").run()
    assert [m.value for m in at.metric if m.label == "유형"] == ["장타형"]
    at.slider(key="iso").set_value(0.30).run()
    assert not at.exception


# ------------------------------------------------------------ CSV 업로드 경로
@pytest.fixture(scope="module")
def good_csv_df():
    from kbo_heatmap.data import load_players, make_demo
    df = make_demo(load_players(), seed=11, pitches_per_pitcher=3000)
    sub = df[(df["batter"] == "김도영") & (df["pitcher"] == "올러")]
    assert len(sub) > 100
    return sub


def _upload(content: bytes, name="pitches.csv"):
    at = _run()
    if not hasattr(at, "file_uploader"):      # 오래된 Streamlit의 테스트 도구는 업로드를 흉내 내지 못한다
        pytest.skip("이 Streamlit 버전의 AppTest는 file_uploader를 지원하지 않습니다")
    at.radio(key="source").set_value("내 CSV 업로드").run()
    at.sidebar.file_uploader(key="upload").upload(name, content, "text/csv").run()
    return at


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp949"])
def test_upload_valid_csv(good_csv_df, encoding):
    at = _upload(good_csv_df.to_csv(index=False).encode(encoding))
    assert not at.exception
    assert not any("DEMO DATA" in w.value for w in at.warning)      # 실제 데이터 모드: 데모 배너 없음
    assert any("불러옴" in s.value for s in at.sidebar.success)
    assert [m.label for m in at.metric] == ["타율", "장타율", "ISO", "유형", "타수"]
    # 데이터가 없는 선수는 목록에서 '(데이터 없음)'으로 표시되고, 선택해도 오류가 나지 않는다
    at.selectbox(key="player_KIA_타자").set_value("나성범").run()
    assert not at.exception
    assert any("데이터에 없어" in i.value or "타석 결과" in i.value for i in at.info)
    at.radio(key="role").set_value("투수").run()
    assert not at.exception and len(at.metric) == 4                 # 올러 투구 있음


@pytest.mark.parametrize("content,expected", [
    (b"a,b\n1,2\n", "필수 컬럼"),
    (b"", "빈 파일"),
    (b"\xff\xfe\xfa\x81\x00\x9d", "인코딩"),
    (b"pitcher,batter,pitch_type,plate_x,plate_z,is_strike,pa_result\n", "남은 투구 데이터가 없습니다"),
])
def test_upload_bad_csv_shows_error_and_falls_back_to_demo(content, expected):
    at = _upload(content)
    assert not at.exception                                            # 앱이 죽지 않는다
    assert any(expected in e.value for e in at.sidebar.error)
    assert any("DEMO DATA" in w.value for w in at.warning)             # 그동안 데모를 보여준다
    assert len(at.metric) == 5


# ------------------------------------------------------------ 분류 탭
def test_classification_tab_demo():
    at = _run()
    assert not any(m.label == "최고 모델" for m in at.metric)       # 버튼 누르기 전에는 학습하지 않는다
    at.button(key="train").click().run()
    assert not at.exception and len(at.error) == 0
    labels = [m.label for m in at.metric]
    assert {"최고 모델", "정확도", "ROC-AUC", "테스트 투구 수"} <= set(labels)
    # 다른 탭의 선수를 바꿔도 결과가 유지된다
    at.radio(key="role").set_value("투수").run()
    assert not at.exception and any(m.label == "최고 모델" for m in at.metric)


def test_classification_tab_too_little_data_shows_error(good_csv_df):
    tiny = good_csv_df.head(60).to_csv(index=False).encode("utf-8-sig")
    at = _upload(tiny)
    assert not at.exception
    at.button(key="train").click().run()
    assert not at.exception
    assert any("너무 적습니다" in e.value for e in at.error)


def test_classification_tab_with_uploaded_data():
    from kbo_heatmap.data import load_players, make_demo
    big = make_demo(load_players(), seed=21, pitches_per_pitcher=500)        # 약 1만 행
    at = _upload(big.to_csv(index=False).encode("utf-8-sig"))
    at.button(key="train").click().run()
    assert not at.exception and len(at.error) == 0
    assert any(m.label == "최고 모델" for m in at.metric)
