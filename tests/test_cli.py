import pandas as pd
import pytest

from kbo_heatmap.cli import main
from kbo_heatmap.data import load_players, make_demo


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "out"
    code = main(["--demo", "--out", str(out), "--teams", "LG,KIWOOM", "--seed", "3"])
    assert code == 0
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    # LG: 타자2+투수2, 키움: 타자0+투수2
    assert figs == ["batter_LG_1.png", "batter_LG_2.png", "pitcher_KIWOOM_1.png",
                    "pitcher_KIWOOM_2.png", "pitcher_LG_1.png", "pitcher_LG_2.png"]
    assert (out / "index.html").read_text(encoding="utf-8").count("DEMO DATA") >= 1
    b = pd.read_csv(out / "batter_summary.csv", encoding="utf-8-sig")
    assert set(b["선수"]) == {"오스틴", "박해민"}
    assert (b["타율"].between(0, 1)).all()
    assert set(b["유형"]) <= {"장타형", "단타형"}


def test_real_csv_path_and_missing_player_is_skipped(tmp_path):
    df = make_demo(load_players(), seed=5, pitches_per_pitcher=3000)
    # 김도영(타자)과 올러(투수)의 맞대결 기록만 남긴다 → 나성범·네일은 데이터가 없다
    sub = df[(df["batter"] == "김도영") & (df["pitcher"] == "올러")]
    assert len(sub) > 100
    csv = tmp_path / "pitches.csv"
    sub.to_csv(csv, index=False, encoding="utf-8-sig")
    out = tmp_path / "out"
    assert main(["--data", str(csv), "--out", str(out), "--teams", "KIA"]) == 0
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "DEMO DATA" not in html.split("<main>")[1].split("<h2>")[0]   # 실제 데이터 모드엔 데모 배너 없음
    assert "나성범" in html and "네일" in html     # 데이터 없는 선수는 '제외' 안내에 표시
    assert (out / "figures" / "batter_KIA_1.png").exists()
    assert (out / "figures" / "pitcher_KIA_1.png").exists()
    assert not (out / "figures" / "batter_KIA_2.png").exists()
    assert not (out / "figures" / "pitcher_KIA_2.png").exists()


def test_error_paths(tmp_path, capsys):
    assert main(["--data", str(tmp_path / "x.csv")]) == 2
    assert main(["--demo", "--teams", "ZZZ", "--out", str(tmp_path)]) == 2
    assert "알 수 없는 구단 코드" in capsys.readouterr().err
    empty = tmp_path / "e.csv"
    pd.DataFrame({"pitcher": ["a"], "batter": ["b"], "pitch_type": ["직구"], "plate_x": [0.0],
                  "plate_z": [2.0], "is_strike": [1], "pa_result": [None]}).to_csv(empty, index=False)
    assert main(["--data", str(empty), "--out", str(tmp_path / "o")]) == 1   # 선수 이름 불일치


def test_grid_3(tmp_path):
    assert main(["--demo", "--grid", "3", "--teams", "KT", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "figures" / "batter_KT_1.png").stat().st_size > 10_000
