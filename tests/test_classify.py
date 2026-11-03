import sys
import types

import numpy as np
import pandas as pd
import pytest

from kbo_heatmap import classify as K
from kbo_heatmap import classify_plots
from kbo_heatmap.classify_cli import main as cls_main
from kbo_heatmap.data import load_players, make_demo
from kbo_heatmap.statcast import convert_statcast, fetch_statcast, load_statcast_csv
from kbo_heatmap.statcast import main as statcast_main

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def demo_df():
    return make_demo(load_players(), seed=3, pitches_per_pitcher=500)


@pytest.fixture(scope="module")
def result(demo_df):
    return K.train_and_evaluate(demo_df, cv=0, rf_trees=40, seed=1)


# ------------------------------------------------------------ 학습/평가
def test_result_structure_and_ranges(result):
    m = result.metrics
    assert K.BASELINE in m.index and len(m) == 5
    assert ((m >= 0) & (m <= 1)).all().all()
    assert result.best_name != K.BASELINE
    assert result.n_train + result.n_test == result.n_rows
    assert result.confusion.sum() == result.n_test
    assert m.loc[K.BASELINE, "ROC-AUC"] == pytest.approx(0.5)          # 아무것도 안 배운 모델
    assert result.baseline_accuracy == pytest.approx(max(result.strike_rate, 1 - result.strike_rate), abs=0.01)


def test_models_beat_baseline_and_feature_engineering_helps(result):
    m = result.metrics
    assert m.loc[result.best_name, "정확도"] > result.baseline_accuracy + 0.05
    # 직선 하나로 사각형 존을 나누기는 어렵다 → 거리 같은 파생변수를 더하면 좋아진다
    assert m.loc["로지스틱 회귀(파생변수)", "ROC-AUC"] > m.loc["로지스틱 회귀(원본 좌표)", "ROC-AUC"] + 0.1


def test_split_is_reproducible_and_seed_matters(demo_df):
    a = K.train_and_evaluate(demo_df, cv=0, rf_trees=20, seed=5)
    b = K.train_and_evaluate(demo_df, cv=0, rf_trees=20, seed=5)
    c = K.train_and_evaluate(demo_df, cv=0, rf_trees=20, seed=6)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
    assert not a.metrics.equals(c.metrics)


def test_cross_validation_columns(demo_df):
    r = K.train_and_evaluate(demo_df.head(3000), cv=3, rf_trees=10)
    assert {"CV정확도(평균)", "CV정확도(표준편차)"} <= set(r.metrics.columns)
    assert r.metrics["CV정확도(평균)"].between(0, 1).all()


def test_max_rows_sampling(demo_df):
    r = K.train_and_evaluate(demo_df, cv=0, rf_trees=10, max_rows=2000)
    assert r.sampled and r.n_rows == 2000


def test_importance_sums_to_one(result):
    assert result.importance.sum() == pytest.approx(1.0)
    assert set(result.importance.index) == set(K.RAW)


# ------------------------------------------------------------ 확률 지도
def test_probability_grid(result):
    xs, zs, p = K.probability_grid(result, "직구", n=30)
    assert p.shape == (30, 30) and (p >= 0).all() and (p <= 1).all()
    center = p[np.argmin(abs(zs - 2.5)), np.argmin(abs(xs - 0.0))]
    corner = p[0, 0]
    assert center > 0.7 and corner < 0.4          # 존 가운데는 스트라이크, 구석(바깥)은 볼
    with pytest.raises(ValueError):
        K.probability_grid(result, "너클볼")


# ------------------------------------------------------------ 입력 검증
def _df(n=300, strike_frac=0.5):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "plate_x": rng.normal(0, .6, n), "plate_z": rng.normal(2.5, .7, n), "pitch_type": "직구",
        "is_strike": (rng.random(n) < strike_frac).astype(int)})


def test_validation_errors():
    with pytest.raises(ValueError, match="너무 적습니다"):
        K.train_and_evaluate(_df(50))
    with pytest.raises(ValueError, match="한 가지"):
        K.train_and_evaluate(_df(300, strike_frac=1.0))
    with pytest.raises(ValueError, match="한쪽이"):
        d = _df(300); d.loc[d.index[:5], "is_strike"] = 0; d.loc[d.index[5:], "is_strike"] = 1
        K.train_and_evaluate(d)
    with pytest.raises(ValueError, match="필요한 컬럼"):
        K.train_and_evaluate(_df().drop(columns=["plate_z"]))
    with pytest.raises(ValueError, match="test_size"):
        K.train_and_evaluate(_df(), test_size=0.9)


def test_nan_rows_are_ignored():
    d = _df(400); d.loc[d.index[:20], "plate_x"] = np.nan
    assert K.train_and_evaluate(d, cv=0, rf_trees=10).n_rows == 380


# ------------------------------------------------------------ 보고서/그림
def test_report_contents(result):
    text = K.build_report(result, demo=True)
    for must in ("DEMO DATA", "이진 분류", "기준선", "혼동행렬", result.best_name, "한계"):
        assert must in text
    assert "DEMO DATA" not in K.build_report(result, demo=False)


def test_plots_written(result, tmp_path):
    classify_plots.plot_probability_map(result, tmp_path / "a.png", demo=True, dpi=60)
    classify_plots.plot_evaluation(result, tmp_path / "b.png", demo=False, dpi=60)
    for f in ("a.png", "b.png"):
        assert (tmp_path / f).read_bytes().startswith(PNG)


# ------------------------------------------------------------ CLI
def test_cli_demo_and_errors(tmp_path, capsys):
    out = tmp_path / "o"
    assert cls_main(["--demo", "--cv", "0", "--out", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"metrics.csv", "probability_map.png", "evaluation.png",
                                               "classification_report.md"}
    assert cls_main(["--data", str(tmp_path / "none.csv")]) == 2
    tiny = tmp_path / "tiny.csv"
    _df(40).assign(pitcher="a", batter="b", pa_result=None).to_csv(tiny, index=False)
    assert cls_main(["--data", str(tiny), "--out", str(tmp_path / "t")]) == 2
    assert "너무 적습니다" in capsys.readouterr().err


# ------------------------------------------------------------ Statcast 변환
def _savant(n=600):
    rng = np.random.default_rng(1)
    x, z = rng.normal(0, .8, n), rng.normal(2.5, .8, n)
    inzone = (abs(x) < .83) & (z > 1.5) & (z < 3.5)
    typ = np.where(rng.random(n) < 0.15, "X", np.where(inzone, "S", "B"))
    return pd.DataFrame({
        "pitch_type": rng.choice(["FF", "SL", "CU", "CH", "SI", "PO"], n),
        "plate_x": x, "plate_z": z, "type": typ,
        "player_name": rng.choice(["Kershaw, Clayton", "Cole, Gerrit"], n),
        "batter": rng.integers(100000, 100050, n)})


def test_convert_statcast_rules():
    raw = _savant()
    out = convert_statcast(raw)
    assert set(out.columns) >= {"pitcher", "batter", "pitch_type", "plate_x", "plate_z", "is_strike", "pa_result"}
    assert set(out["pitch_type"]) == {"직구", "변화구"}
    n_bs = raw["type"].isin(["B", "S"]).sum()
    n_po = (raw["type"].isin(["B", "S"]) & (raw["pitch_type"] == "PO")).sum()
    assert len(out) == n_bs - n_po                          # 인플레이(X)와 알 수 없는 구종(PO)은 제외
    first = raw[raw["type"].isin(["B", "S"]) & (raw["pitch_type"] != "PO")].iloc[0]
    assert out.iloc[0]["is_strike"] == int(first["type"] == "S")


def test_convert_statcast_errors():
    with pytest.raises(ValueError, match="Statcast 형식"):
        convert_statcast(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="하나도 없습니다"):
        convert_statcast(_savant().assign(type="X"))


def test_statcast_roundtrip_and_classification(tmp_path):
    src = tmp_path / "savant.csv"
    _savant(3000).to_csv(src, index=False)
    out = tmp_path / "conv.csv"
    assert statcast_main(["--from-csv", str(src), "--out", str(out)]) == 0
    assert cls_main(["--data", str(out), "--cv", "0", "--out", str(tmp_path / "res")]) == 0
    r = K.train_and_evaluate(load_statcast_csv(src), cv=0, rf_trees=30)
    assert r.metrics.loc[r.best_name, "정확도"] > 0.9       # 존 규칙으로 만든 라벨이라 거의 완벽히 맞혀야 정상


def test_statcast_cli_errors(tmp_path, capsys):
    assert statcast_main(["--from-csv", str(tmp_path / "x.csv"), "--out", str(tmp_path / "o.csv")]) == 2
    with pytest.raises(SystemExit) as e:                       # --start만 주고 --end를 빼먹은 경우
        statcast_main(["--start", "2026-06-01", "--out", str(tmp_path / "o.csv")])
    assert e.value.code == 2
    assert "--end" in capsys.readouterr().err


def test_fetch_validation_and_failure_messages(monkeypatch):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        fetch_statcast("06/01/2026", "2026-06-02")
    with pytest.raises(ValueError, match="빠릅니다"):
        fetch_statcast("2026-06-05", "2026-06-01")
    with pytest.raises(ValueError, match="너무 깁니다"):
        fetch_statcast("2026-01-01", "2026-06-01")
    monkeypatch.setitem(sys.modules, "pybaseball", None)       # 설치되지 않은 상태를 흉내
    with pytest.raises(RuntimeError, match="pip install pybaseball"):
        fetch_statcast("2026-06-01", "2026-06-02")

    fake = types.ModuleType("pybaseball")

    def boom(**kw):
        raise OSError("network down")
    fake.statcast = boom
    monkeypatch.setitem(sys.modules, "pybaseball", fake)
    with pytest.raises(RuntimeError, match="--from-csv"):       # 실패 시 대안 안내
        fetch_statcast("2026-06-01", "2026-06-02")

    fake.statcast = lambda **kw: _savant()
    assert len(fetch_statcast("2026-06-01", "2026-06-02")) > 100
    fake.statcast = lambda **kw: pd.DataFrame()
    with pytest.raises(RuntimeError, match="받아온 데이터가 없습니다"):
        fetch_statcast("2026-06-01", "2026-06-02")
