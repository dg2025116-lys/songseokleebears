import pandas as pd
import pytest

from kbo_heatmap import render
from kbo_heatmap.data import load_pitches_bytes, load_players, make_demo
from kbo_heatmap.tables import summary_tables

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def demo():
    teams = load_players()
    return teams, make_demo(teams, seed=7, pitches_per_pitcher=800)


def test_pngs_are_valid(demo):
    _, df = demo
    png, s = render.batter_png(df, "김도영", "KIA 타이거즈", demo=True)
    assert png.startswith(PNG) and s["name"] == "김도영"
    png, s = render.pitcher_png(df, "곽빈", "두산 베어스", grid=3, demo=True)
    assert png.startswith(PNG) and s["pitches"] > 0
    assert render.hero_png(demo=True).startswith(PNG)


def test_missing_player_returns_none(demo):
    _, df = demo
    assert render.batter_png(df, "없는타자", "x") == (None, None)
    assert render.pitcher_png(df, "없는투수", "x") == (None, None)


def test_summary_tables(demo):
    teams, df = demo
    bt, pt = summary_tables(df, teams)
    assert len(bt) == 18 and len(pt) == 20
    assert {"선수", "타율", "유형"} <= set(bt.columns)
    assert bt["타율"].between(0, 1).all()


def test_load_pitches_bytes_errors():
    with pytest.raises(ValueError, match="빈 파일"):
        load_pitches_bytes(b"")
    with pytest.raises(ValueError, match="인코딩"):
        load_pitches_bytes(b"\xff\xfe\xfa\x81\x00\x9d")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_pitches_bytes("a,b\n1,2\n".encode())
    header_only = "pitcher,batter,pitch_type,plate_x,plate_z,is_strike,pa_result\n".encode()
    with pytest.raises(ValueError):
        load_pitches_bytes(header_only)


def test_summary_tables_have_numeric_columns_for_streamlit(demo):
    """빈 칸이 ''(문자열)로 섞이면 Streamlit의 Arrow 변환이 실패하므로 숫자 컬럼은 숫자로만 채운다."""
    teams, df = demo
    bt, pt = summary_tables(df, teams)
    for col in ("시즌타율(참고)", "시즌OPS(참고)", "장타비율(안타 중)"):
        assert pd.api.types.is_float_dtype(bt[col]), col
    assert pd.api.types.is_float_dtype(pt["시즌ERA(참고)"])
    import pyarrow as pa
    pa.Table.from_pandas(bt)
    pa.Table.from_pandas(pt)
