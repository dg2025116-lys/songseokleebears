"""요약표의 행 만들기. CLI(CSV 저장)와 Streamlit 앱(화면 표)이 같은 함수를 씁니다."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from . import config as C
from . import metrics as M
from .data import Player, Team

NAN = float("nan")   # 빈 칸은 ''가 아니라 NaN으로 둬야 숫자 컬럼이 유지된다(Streamlit/Arrow 변환 오류 방지)


def batter_row(team_name: str, player: Player, s: Dict) -> Dict:
    return {
        "구단": team_name, "선수": s["name"], "타석": s["pa"], "타수": s["ab"],
        "안타": s["hits"], "홈런": s["hr"], "타율": round(s["avg"], 3),
        "장타율": round(s["slg"], 3), "ISO": round(s["iso"], 3),
        "장타비율(안타 중)": round(s["xbh_ratio"], 3),
        "유형": s["type"], "표본충분": "예" if s["sample_ok"] else "아니오",
        "강한존": s["strong_zone"], "약한존": s["weak_zone"],
        "시즌타율(참고)": player.ref.get("avg", NAN), "시즌OPS(참고)": player.ref.get("ops", NAN),
        "2026기록확인": "예" if player.verified else "아니오"}


def pitcher_row(team_name: str, player: Player, s: Dict) -> Dict:
    return {
        "구단": team_name, "선수": s["name"], "투구수": s["pitches"],
        "직구비율": round(s["fb_share"], 3), "직구스트라이크율": round(s["fb_strike"], 3),
        "변화구스트라이크율": round(s["br_strike"], 3),
        "직구존내비율": round(s["fb_zone_pct"], 3), "변화구존내비율": round(s["br_zone_pct"], 3),
        "직구잘넣는존": s["fb_best_zone"], "변화구잘넣는존": s["br_best_zone"],
        "시즌ERA(참고)": player.ref.get("era", NAN), "2026기록확인": "예" if player.verified else "아니오"}


def summary_tables(df: pd.DataFrame, teams: List[Team], grid: int = C.DEFAULT_GRID,
                   iso_threshold: float = C.ISO_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """설정된 모든 선수의 요약표(타자, 투수). 데이터가 없는 선수는 빠집니다."""
    brows, prows = [], []
    for t in teams:
        for p in t.batters:
            s = M.batter_summary(df, p.name, grid, iso_threshold)
            if s is not None:
                brows.append(batter_row(t.name, p, s))
        for p in t.pitchers:
            s = M.pitcher_summary(df, p.name, grid)
            if s is not None:
                prows.append(pitcher_row(t.name, p, s))
    return pd.DataFrame(brows), pd.DataFrame(prows)
