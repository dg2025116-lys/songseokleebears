import json

import numpy as np
import pandas as pd
import pytest

from kbo_heatmap import config as C
from kbo_heatmap.data import (load_players, load_pitches, make_demo, normalize_pitch_type,
                              prepare_pitches, zone_to_xy)


def test_default_players_file_is_valid():
    teams = load_players()
    assert len(teams) == 10
    assert sum(len(t.batters) for t in teams) == 18          # 키움은 대표 타자 없음
    assert sum(len(t.pitchers) for t in teams) == 20
    kiwoom = next(t for t in teams if t.code == "KIWOOM")
    assert kiwoom.batters == []
    assert {p.name for p in kiwoom.pitchers} == {"알칸타라", "안우진"}


def test_duplicate_player_names_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"teams": [
        {"code": "A", "name": "A", "batters": [{"name": "갑"}], "pitchers": []},
        {"code": "B", "name": "B", "batters": [{"name": "갑"}], "pitchers": []}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        load_players(f)


def test_missing_players_file():
    with pytest.raises(FileNotFoundError):
        load_players("nope.json")


@pytest.mark.parametrize("raw,expected", [
    ("직구", "직구"), ("ff", "직구"), (" 포심 ", "직구"),
    ("슬라이더", "변화구"), ("CU", "변화구"), ("체인지업", "변화구"),
    ("???", None), (None, None), (float("nan"), None)])
def test_normalize_pitch_type(raw, expected):
    assert normalize_pitch_type(raw) == expected


def test_zone_to_xy_corners():
    x, z = zone_to_xy("A1")          # 왼쪽 위
    assert x < 0 and z > 3.4
    x, z = zone_to_xy("E5")          # 오른쪽 아래
    assert x > 0 and z < 1.6
    x, z = zone_to_xy("c3")          # 가운데
    assert abs(x) < 1e-9 and abs(z - 2.5) < 1e-9
    with pytest.raises(ValueError):
        zone_to_xy("Z9")


def _base_df(**over):
    d = {"pitcher": ["투"] * 3, "batter": ["타"] * 3, "pitch_type": ["직구", "SL", "??"],
         "plate_x": [0.0, 0.5, 0.1], "plate_z": [2.5, 2.0, 2.0],
         "is_strike": [1, 0, 1], "pa_result": ["1루타", None, ""]}
    d.update(over)
    return pd.DataFrame(d)


def test_prepare_drops_bad_rows_and_normalizes():
    out = prepare_pitches(_base_df())
    assert len(out) == 2                                   # '??' 구종 행 제거
    assert list(out["pitch_type"]) == ["직구", "변화구"]
    assert out["pa_result"].notna().sum() == 1
    assert out["is_strike"].dtype.kind == "i"


def test_prepare_accepts_bool_and_text_strike():
    out = prepare_pitches(_base_df(pitch_type=["직구"] * 3, is_strike=[True, False, True]))
    assert list(out["is_strike"]) == [1, 0, 1]
    out = prepare_pitches(_base_df(pitch_type=["직구"] * 3, is_strike=["Y", "n", "1"]))
    assert list(out["is_strike"]) == [1, 0, 1]


def test_prepare_missing_columns():
    with pytest.raises(ValueError, match="필수 컬럼"):
        prepare_pitches(pd.DataFrame({"pitcher": ["a"]}))
    df = _base_df().drop(columns=["plate_x", "plate_z"])
    with pytest.raises(ValueError, match="위치"):
        prepare_pitches(df)


def test_prepare_uses_zone_codes_when_no_coordinates():
    df = _base_df(pitch_type=["직구"] * 3).drop(columns=["plate_x", "plate_z"])
    df["zone"] = ["C3", "A1", "E5"]
    out = prepare_pitches(df)
    assert out["plate_x"].between(*C.X_RANGE).all()
    assert out["plate_z"].between(*C.Z_RANGE).all()


def test_prepare_is_idempotent():
    once = prepare_pitches(_base_df(pitch_type=["직구"] * 3))
    twice = prepare_pitches(once)
    assert once.equals(twice)


def test_prepare_all_rows_invalid():
    with pytest.raises(ValueError, match="남은 투구 데이터가 없습니다"):
        prepare_pitches(_base_df(pitch_type=["??"] * 3))


def test_load_pitches_cp949_and_utf8(tmp_path):
    df = _base_df(pitch_type=["직구"] * 3)
    for enc in ("utf-8-sig", "cp949"):
        f = tmp_path / f"{enc}.csv"
        df.to_csv(f, index=False, encoding=enc)
        assert len(load_pitches(f)) == 3


def test_demo_is_reproducible_and_covers_players():
    teams = load_players()
    a = make_demo(teams, seed=1, pitches_per_pitcher=300)
    b = make_demo(teams, seed=1, pitches_per_pitcher=300)
    assert a.equals(b)
    assert set(a["pitcher"]) == {p.name for t in teams for p in t.pitchers}
    assert a["is_strike"].isin([0, 1]).all()
    assert set(a["pitch_type"]) == {"직구", "변화구"}
