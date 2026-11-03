import numpy as np
import pandas as pd
import pytest

from kbo_heatmap import metrics as M
from kbo_heatmap.data import prepare_pitches


def _ab_df(results, x=0.0, z=2.5, batter="타자"):
    return prepare_pitches(pd.DataFrame({
        "pitcher": "투수", "batter": batter, "pitch_type": "직구",
        "plate_x": x, "plate_z": z, "is_strike": 1, "pa_result": results}))


def test_batting_average_and_iso_by_hand():
    # 10타수: 1루타3, 2루타1, 홈런1, 아웃5 (+ 볼넷 2개는 타수 제외)
    res = ["1루타"] * 3 + ["2루타"] + ["홈런"] + ["아웃"] * 5 + ["볼넷"] * 2
    s = M.batter_summary(_ab_df(res), "타자")
    assert s["ab"] == 10 and s["pa"] == 12
    assert s["hits"] == 5 and s["hr"] == 1 and s["xbh"] == 2
    assert s["avg"] == pytest.approx(0.5)
    assert s["slg"] == pytest.approx((3 + 2 + 4) / 10)          # 0.9
    assert s["iso"] == pytest.approx(0.4)
    assert s["xbh_ratio"] == pytest.approx(2 / 5)


def test_walks_hbp_sacrifice_not_counted_as_at_bats():
    res = ["볼넷", "사구", "희생플라이", "희생번트", "고의4구", "삼진", "1루타"]
    s = M.batter_summary(_ab_df(res), "타자")
    assert s["ab"] == 2 and s["hits"] == 1


def test_classification_threshold_and_sample_flag():
    power = _ab_df(["홈런"] * 4 + ["아웃"] * 6)
    contact = _ab_df(["1루타"] * 4 + ["아웃"] * 6)
    assert M.batter_summary(power, "타자")["type"] == "장타형"
    assert M.batter_summary(contact, "타자")["type"] == "단타형"
    assert M.batter_summary(contact, "타자", min_ab_classify=50)["sample_ok"] is False
    assert M.batter_summary(contact, "타자", min_ab_classify=5)["sample_ok"] is True
    # 임계값을 높이면 같은 선수가 단타형이 된다
    assert M.batter_summary(power, "타자", iso_threshold=9.0)["type"] == "단타형"


def test_no_at_bats_returns_none():
    assert M.batter_summary(_ab_df(["볼넷"]), "타자") is None
    assert M.batter_summary(_ab_df(["1루타"]), "다른사람") is None


def test_grid_binning_edges_and_clipping():
    df = pd.DataFrame({"plate_x": [-1.5, 1.5, -9, 9, 0.0], "plate_z": [1.0, 4.0, -9, 9, 2.5]})
    g = M.add_grid(df, 5)
    assert list(g["gx"]) == [0, 4, 0, 4, 2]
    assert list(g["gz"]) == [0, 4, 0, 4, 2]
    with pytest.raises(ValueError):
        M.make_edges(4)


def test_batter_zone_grid_puts_hits_in_right_cell():
    # 가운데(2,2)에서 4타수 2안타, 오른쪽 위(4,4)에서 3타수 0안타
    mid = _ab_df(["1루타", "1루타", "아웃", "아웃"], x=0.0, z=2.5)
    corner = _ab_df(["아웃"] * 3, x=1.4, z=3.9)
    df = pd.concat([mid, corner], ignore_index=True)
    rate, cnt = M.batter_zone_grid(df, "타자", grid=5, min_ab_zone=3)
    assert cnt[2, 2] == 4 and rate[2, 2] == pytest.approx(0.5)
    assert cnt[4, 4] == 3 and rate[4, 4] == pytest.approx(0.0)
    assert np.isnan(rate[0, 0]) and cnt[0, 0] == 0


def test_min_sample_hides_small_cells():
    df = _ab_df(["1루타", "1루타"], x=0.0, z=2.5)
    rate, cnt = M.batter_zone_grid(df, "타자", grid=5, min_ab_zone=3)
    assert cnt[2, 2] == 2 and np.isnan(rate[2, 2])


def test_best_cell_respects_min_sample_and_ties():
    rate = np.array([[0.9, np.nan], [0.5, 0.5]])
    cnt = np.array([[2, 0], [10, 20]])
    assert M.best_cell(rate, cnt, "max", min_n=5) == (1, 1)      # 0.9 칸은 표본 부족, 동률은 표본 많은 쪽
    assert M.best_cell(rate, cnt, "max", min_n=1) == (0, 0)
    assert M.best_cell(rate, cnt, "min", min_n=1) == (1, 1)
    assert M.best_cell(np.full((2, 2), np.nan), np.zeros((2, 2), int)) is None


def test_zone_label():
    assert M.zone_label((0, 0), 5) == "아주 낮음·왼쪽 밖"
    assert M.zone_label((2, 2), 5) == "중간·가운데"
    assert M.zone_label((1, 1), 3) == "중간·가운데"
    assert M.zone_label((2, 2), 3) == "높음·오른쪽"
    assert M.zone_label(None, 5) == "-"


def test_pitcher_summary_strike_rates():
    rows = []
    for _ in range(30):   # 직구: 가운데 30개, 스트라이크 24개
        rows.append(("투수", "직구", 0.0, 2.5, 1 if len(rows) < 24 else 0))
    for _ in range(10):   # 변화구: 낮은 곳 10개, 스트라이크 2개
        rows.append(("투수", "변화구", 0.0, 1.1, 1 if len(rows) < 32 else 0))
    df = prepare_pitches(pd.DataFrame(rows, columns=["pitcher", "pitch_type", "plate_x", "plate_z", "is_strike"])
                         .assign(batter="타자", pa_result=None))
    s = M.pitcher_summary(df, "투수", grid=5, min_pitch=5, min_pitch_best=5)
    assert s["pitches"] == 40
    assert s["fb_share"] == pytest.approx(0.75)
    assert s["fb_strike"] == pytest.approx(24 / 30)
    assert s["br_strike"] == pytest.approx(2 / 10)
    assert s["fb_best_cell"] == (2, 2)
    assert s["fb_zone_pct"] == pytest.approx(1.0)       # 전부 스트라이크존 안
    assert s["br_zone_pct"] == pytest.approx(0.0)       # 전부 존 아래
    assert M.pitcher_summary(df, "없는투수") is None
