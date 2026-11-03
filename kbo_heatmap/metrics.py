"""타자/투수 지표 계산. 그림이나 파일 입출력과 무관한 순수 계산만 담았습니다."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C

Cell = Tuple[int, int]  # (gz, gx): gz 0=아래, gx 0=포수 시점 왼쪽


# ============================================================ 격자
def make_edges(grid: int):
    if grid not in C.SUPPORTED_GRIDS:
        raise ValueError(f"grid는 {C.SUPPORTED_GRIDS} 중 하나여야 합니다: {grid}")
    return np.linspace(*C.X_RANGE, grid + 1), np.linspace(*C.Z_RANGE, grid + 1)


def add_grid(df: pd.DataFrame, grid: int) -> pd.DataFrame:
    """plate_x/plate_z를 격자 번호 gx, gz로 바꿉니다. 범위 밖 투구는 가장자리 칸에 포함됩니다."""
    xe, ze = make_edges(grid)
    out = df.dropna(subset=["plate_x", "plate_z"]).copy()
    out["gx"] = np.clip(np.digitize(out["plate_x"].to_numpy(), xe) - 1, 0, grid - 1)
    out["gz"] = np.clip(np.digitize(out["plate_z"].to_numpy(), ze) - 1, 0, grid - 1)
    return out


def grid_stats(gx: np.ndarray, gz: np.ndarray, values: np.ndarray, grid: int, min_n: int):
    """칸별 평균(표본이 min_n 미만이면 NaN)과 표본 수를 반환합니다."""
    cnt = np.zeros((grid, grid), dtype=int)
    tot = np.zeros((grid, grid), dtype=float)
    np.add.at(cnt, (gz, gx), 1)
    np.add.at(tot, (gz, gx), values)
    rate = np.where(cnt >= min_n, tot / np.maximum(cnt, 1), np.nan)
    return rate, cnt


def best_cell(rate: np.ndarray, cnt: np.ndarray, kind: str = "max",
              min_n: int = 1) -> Optional[Cell]:
    """값이 가장 높은(max) 또는 낮은(min) 칸. 표본이 min_n 미만인 칸은 제외하고,
    동률이면 표본이 많은 칸을 고릅니다. 후보가 없으면 None."""
    ok = ~np.isnan(rate) & (cnt >= min_n)
    cells = [(gz, gx) for gz, gx in zip(*np.where(ok))]
    if not cells:
        return None
    sign = -1 if kind == "max" else 1
    cells.sort(key=lambda c: (sign * rate[c], -cnt[c]))
    return int(cells[0][0]), int(cells[0][1])


_X_LABELS = {5: ["왼쪽 밖", "왼쪽", "가운데", "오른쪽", "오른쪽 밖"], 3: ["왼쪽", "가운데", "오른쪽"]}
_Z_LABELS = {5: ["아주 낮음", "낮음", "중간", "높음", "아주 높음"], 3: ["낮음", "중간", "높음"]}


def zone_label(cell: Optional[Cell], grid: int) -> str:
    """칸을 '높이·좌우(포수 시점)' 형태의 한글로. 예: '낮음·오른쪽'"""
    if cell is None:
        return "-"
    gz, gx = cell
    return f"{_Z_LABELS[grid][gz]}·{_X_LABELS[grid][gx]}"


def in_strike_zone(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    x0, x1, z0, z1 = C.STRIKE_ZONE
    return (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)


# ============================================================ 타자
def _at_bats(df: pd.DataFrame, batter: str) -> pd.DataFrame:
    sub = df[(df["batter"] == batter) & df["pa_result"].notna()]
    return sub[~sub["pa_result"].isin(C.NOT_AB)]


def batter_summary(df: pd.DataFrame, batter: str, grid: int = C.DEFAULT_GRID,
                   iso_threshold: float = C.ISO_THRESHOLD,
                   min_ab_classify: int = C.MIN_AB_CLASSIFY,
                   min_ab_zone: int = C.MIN_AB_ZONE,
                   min_ab_best: int = C.MIN_AB_BEST) -> Optional[Dict]:
    """타율·장타율·ISO·장타형/단타형·강한/약한 존. 타수가 0이면 None."""
    pa_rows = df[(df["batter"] == batter) & df["pa_result"].notna()]
    ab = _at_bats(df, batter)
    n_ab = len(ab)
    if n_ab == 0:
        return None
    bases = ab["pa_result"].map(C.HIT_BASES).fillna(0)
    hits = int((bases > 0).sum())
    total_bases = int(bases.sum())
    avg, slg = hits / n_ab, total_bases / n_ab
    iso = slg - avg
    xbh = int((bases >= 2).sum())
    rate, cnt = batter_zone_grid(df, batter, grid, min_ab_zone)
    strong = best_cell(rate, cnt, "max", min_ab_best)
    weak = best_cell(rate, cnt, "min", min_ab_best)
    return {
        "name": batter,
        "pa": int(len(pa_rows)), "ab": n_ab, "hits": hits,
        "hr": int((bases == 4).sum()), "xbh": xbh,
        "avg": avg, "slg": slg, "iso": iso,
        "xbh_ratio": xbh / hits if hits else float("nan"),
        "type": "장타형" if iso >= iso_threshold else "단타형",
        "sample_ok": n_ab >= min_ab_classify,
        "strong_cell": strong, "weak_cell": weak,
        "strong_zone": zone_label(strong, grid),
        "weak_zone": zone_label(weak, grid),
    }


def batter_zone_grid(df: pd.DataFrame, batter: str, grid: int = C.DEFAULT_GRID,
                     min_ab_zone: int = C.MIN_AB_ZONE):
    """칸별 타율 행렬과 타수 행렬. 타석을 끝낸 투구의 위치를 기준으로 합니다."""
    ab = add_grid(_at_bats(df, batter), grid)
    hit = ab["pa_result"].isin(C.HIT_BASES).to_numpy(dtype=float)
    return grid_stats(ab["gx"].to_numpy(), ab["gz"].to_numpy(), hit, grid, min_ab_zone)


# ============================================================ 투수
def pitcher_zone_grid(df: pd.DataFrame, pitcher: str, pitch_type: str,
                      grid: int = C.DEFAULT_GRID, min_pitch: int = C.MIN_PITCH_ZONE):
    """칸별 스트라이크 비율 행렬, 투구 수 행렬, 해당 구종 총 투구 수."""
    p = add_grid(df[(df["pitcher"] == pitcher) & (df["pitch_type"] == pitch_type)], grid)
    rate, cnt = grid_stats(p["gx"].to_numpy(), p["gz"].to_numpy(),
                           p["is_strike"].to_numpy(dtype=float), grid, min_pitch)
    return rate, cnt, len(p)


def pitcher_summary(df: pd.DataFrame, pitcher: str, grid: int = C.DEFAULT_GRID,
                    min_pitch: int = C.MIN_PITCH_ZONE,
                    min_pitch_best: int = C.MIN_PITCH_BEST) -> Optional[Dict]:
    p = df[df["pitcher"] == pitcher]
    if p.empty:
        return None
    out: Dict = {"name": pitcher, "pitches": int(len(p))}
    for ptype, key in (("직구", "fb"), ("변화구", "br")):
        sub = p[p["pitch_type"] == ptype]
        n = len(sub)
        rate, cnt, _ = pitcher_zone_grid(df, pitcher, ptype, grid, min_pitch)
        best = best_cell(rate, cnt, "max", min_pitch_best)
        out[f"{key}_n"] = int(n)
        out[f"{key}_share"] = n / len(p)
        out[f"{key}_strike"] = float(sub["is_strike"].mean()) if n else float("nan")
        out[f"{key}_zone_pct"] = (float(in_strike_zone(sub["plate_x"].to_numpy(),
                                                       sub["plate_z"].to_numpy()).mean())
                                  if n else float("nan"))
        out[f"{key}_best_cell"] = best
        out[f"{key}_best_zone"] = zone_label(best, grid)
        out[f"{key}_best_rate"] = float(rate[best]) if best else float("nan")
    return out
