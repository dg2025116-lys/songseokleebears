"""MLB Statcast(Baseball Savant) 투구 데이터를 이 프로젝트의 CSV 형식으로 바꿉니다.

왜 필요한가: KBO는 투구 위치 데이터를 공개하지 않지만, MLB Statcast는 누구나 내려받을 수 있는 실제 투구
위치 데이터(plate_x, plate_z)입니다. 분류 모델 학습에는 '위치 + 스트라이크/볼 라벨'만 있으면 되므로
Statcast로 실제 데이터 분석을 할 수 있습니다.

사용법 (둘 중 하나)
  # 1) Baseball Savant 검색 페이지에서 직접 받은 CSV를 변환 (가장 확실한 방법)
  python -m kbo_heatmap.statcast --from-csv savant.csv --out data/pitches_statcast.csv

  # 2) pybaseball로 기간을 지정해 내려받기 (pip install pybaseball 필요, 인터넷 필요)
  python -m kbo_heatmap.statcast --start 2026-06-01 --end 2026-06-07 --out data/pitches_statcast.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .data import prepare_pitches

logger = logging.getLogger(__name__)
REQUIRED = ("pitch_type", "plate_x", "plate_z", "type")


def convert_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    """Statcast 투구표를 프로젝트 형식으로 변환합니다.

    - 공이 볼(type='B')이면 is_strike=0, 스트라이크(type='S': 루킹·헛스윙·파울)이면 1
    - 인플레이(type='X')는 볼도 스트라이크도 아니므로 제외
    - 투수 이름은 player_name(없으면 pitcher ID), 타자는 batter ID
    - 타석 결과(pa_result)는 채우지 않음 → 이 데이터는 분류 모델용
    """
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Statcast 형식이 아닙니다. 없는 컬럼: {missing} "
                         f"(필요: {list(REQUIRED)})")
    df = raw[raw["type"].isin(["B", "S"])]
    if df.empty:
        raise ValueError("볼(B)/스트라이크(S) 투구가 하나도 없습니다. 올바른 Statcast 파일인지 확인하세요.")
    pitcher = df["player_name"] if "player_name" in df.columns else df.get("pitcher", pd.Series("unknown", index=df.index))
    batter = df["batter"] if "batter" in df.columns else pd.Series("unknown", index=df.index)
    out = pd.DataFrame({
        "pitcher": pitcher.astype(str).to_numpy(),
        "batter": batter.astype(str).to_numpy(),
        "pitch_type": df["pitch_type"].to_numpy(),
        "plate_x": pd.to_numeric(df["plate_x"], errors="coerce").to_numpy(),
        "plate_z": pd.to_numeric(df["plate_z"], errors="coerce").to_numpy(),
        "is_strike": (df["type"] == "S").astype(int).to_numpy(),
        "pa_result": np.nan,
    })
    return prepare_pitches(out)


def load_statcast_csv(path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}") from None
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise ValueError(f"CSV를 읽을 수 없습니다: {e}") from None
    return convert_statcast(raw)


def fetch_statcast(start: str, end: str) -> pd.DataFrame:
    """pybaseball로 기간 내 Statcast 데이터를 받아 변환합니다(인터넷과 pybaseball 필요)."""
    try:
        d0, d1 = (datetime.strptime(x, "%Y-%m-%d") for x in (start, end))
    except ValueError:
        raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다. 예: 2026-06-01") from None
    if d1 < d0:
        raise ValueError("종료일이 시작일보다 빠릅니다.")
    if (d1 - d0).days > 14:
        raise ValueError("기간이 너무 깁니다(최대 15일). 데이터가 매우 커져서 실패하기 쉽습니다.")
    try:
        from pybaseball import statcast
    except ImportError:
        raise RuntimeError("pybaseball이 설치되어 있지 않습니다: pip install pybaseball "
                           "(또는 --from-csv로 Baseball Savant에서 받은 CSV를 변환하세요)") from None
    try:
        raw = statcast(start_dt=start, end_dt=end)
    except Exception as e:  # 네트워크/사이트 변경 등
        raise RuntimeError(f"Statcast 다운로드에 실패했습니다: {e}\n"
                           "→ 대신 baseballsavant.mlb.com 의 Statcast Search에서 CSV를 직접 받아 "
                           "--from-csv 로 변환하세요.") from None
    if raw is None or len(raw) == 0:
        raise RuntimeError("해당 기간에 받아온 데이터가 없습니다(경기가 없는 기간일 수 있습니다).")
    return convert_statcast(raw)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="kbo_heatmap.statcast", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-csv", metavar="CSV", help="Baseball Savant에서 받은 CSV")
    src.add_argument("--start", metavar="YYYY-MM-DD", help="pybaseball로 받을 시작일 (--end 필요)")
    ap.add_argument("--end", metavar="YYYY-MM-DD")
    ap.add_argument("--out", required=True, metavar="CSV", help="변환된 CSV 저장 경로")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        if args.from_csv:
            df = load_statcast_csv(args.from_csv)
        else:
            if not args.end:
                ap.error("--start를 쓸 때는 --end도 필요합니다.")
            df = fetch_statcast(args.start, args.end)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {args.out}  ({len(df):,}구, 스트라이크 비율 {df['is_strike'].mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
