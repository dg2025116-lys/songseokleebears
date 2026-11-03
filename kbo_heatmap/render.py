"""그림을 파일이 아닌 PNG bytes로 만들어 줍니다(웹앱용). matplotlib은 스레드에 안전하지 않아 잠금을 씁니다."""
from __future__ import annotations

import io
import threading
from typing import Dict, Optional, Tuple

import pandas as pd

from . import classify_plots
from . import config as C
from . import metrics as M
from . import plots, stadium

_LOCK = threading.Lock()
WEB_DPI = 110


def batter_png(df: pd.DataFrame, name: str, team_name: str, grid: int = C.DEFAULT_GRID,
               iso_threshold: float = C.ISO_THRESHOLD,
               demo: bool = False) -> Tuple[Optional[bytes], Optional[Dict]]:
    """(PNG bytes, 요약 dict). 타수가 없으면 (None, None)."""
    summ = M.batter_summary(df, name, grid, iso_threshold)
    if summ is None:
        return None, None
    with _LOCK:
        buf = io.BytesIO()
        plots.plot_batter(df, name, team_name, summ, buf, grid=grid, demo=demo, dpi=WEB_DPI)
    return buf.getvalue(), summ


def pitcher_png(df: pd.DataFrame, name: str, team_name: str, grid: int = C.DEFAULT_GRID,
                demo: bool = False) -> Tuple[Optional[bytes], Optional[Dict]]:
    summ = M.pitcher_summary(df, name, grid)
    if summ is None:
        return None, None
    with _LOCK:
        buf = io.BytesIO()
        plots.plot_pitcher(df, name, team_name, summ, buf, grid=grid, demo=demo, dpi=WEB_DPI)
    return buf.getvalue(), summ


def hero_png(demo: bool = False) -> bytes:
    with _LOCK:
        buf = io.BytesIO()
        stadium.save_hero(buf, "KBO 대표 선수 분석", "타율 · 장타 유형 · 존별 히트맵", demo=demo)
    return buf.getvalue()


def probability_map_png(result, demo: bool = False) -> bytes:
    with _LOCK:
        buf = io.BytesIO()
        classify_plots.plot_probability_map(result, buf, demo=demo, dpi=WEB_DPI)
    return buf.getvalue()


def evaluation_png(result, demo: bool = False) -> bytes:
    with _LOCK:
        buf = io.BytesIO()
        classify_plots.plot_evaluation(result, buf, demo=demo, dpi=WEB_DPI)
    return buf.getvalue()
