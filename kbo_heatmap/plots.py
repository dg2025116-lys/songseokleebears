"""타자/투수 히트맵 그림. 배경은 stadium.add_background(잠실의 밤)를 사용합니다."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from . import config as C
from . import metrics as M
from . import stadium

logger = logging.getLogger(__name__)

FONT_CANDIDATES = (
    "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "NanumBarunGothic",
    "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP", "Noto Sans CJK SC",
    "Source Han Sans K", "UnDotum",
)
PANEL = (1, 1, 1, 0.90)


def setup_font(font_path: Optional[str] = None) -> Optional[str]:
    """한글이 깨지지 않도록 폰트를 고릅니다. 찾지 못하면 경고만 하고 기본 폰트를 씁니다."""
    from matplotlib import font_manager as fm
    chosen = None
    if font_path:
        fm.fontManager.addfont(font_path)
        chosen = fm.FontProperties(fname=font_path).get_name()
    else:
        installed = {f.name for f in fm.fontManager.ttflist}
        chosen = next((c for c in FONT_CANDIDATES if c in installed), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    else:
        logger.warning("한글 폰트를 찾지 못했습니다. 글자가 깨지면 --font-path로 .ttf/.otf 파일을 지정하세요.")
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ------------------------------------------------------------ 공통
def _draw_heat(ax, mat, cnt, cmap_name, vmin, vmax, fmt, highlight=None, grid=C.DEFAULT_GRID):
    cmap = plt.get_cmap(cmap_name).with_extremes(bad="#d5dbe3")
    norm = Normalize(vmin, vmax)
    im = ax.imshow(np.ma.masked_invalid(mat), origin="lower", cmap=cmap,
                   norm=norm, extent=[*C.X_RANGE, *C.Z_RANGE], aspect="auto")
    xe, ze = M.make_edges(grid)
    for gz in range(grid):
        for gx in range(grid):
            n = int(cnt[gz, gx])
            if n == 0:
                continue
            cx, cz = (xe[gx] + xe[gx + 1]) / 2, (ze[gz] + ze[gz + 1]) / 2
            if np.isnan(mat[gz, gx]):
                ax.text(cx, cz, f"n={n}", ha="center", va="center", fontsize=7, color="#7a8696")
            else:
                r, g, b, _ = cmap(norm(mat[gz, gx]))
                dark_cell = (0.299 * r + 0.587 * g + 0.114 * b) < 0.45
                ax.text(cx, cz, f"{fmt.format(mat[gz, gx])}\nn={n}", ha="center", va="center",
                        fontsize=8, color="white" if dark_cell else "#102030", linespacing=1.15)
    x0, x1, z0, z1 = C.STRIKE_ZONE
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="#0b1b2e", lw=2.4, zorder=5))
    if highlight is not None:
        gz, gx = highlight
        ax.add_patch(Rectangle((xe[gx], ze[gz]), xe[gx + 1] - xe[gx], ze[gz + 1] - ze[gz],
                               fill=False, ec=stadium.GOLD, lw=3.2, zorder=6))
    ax.set_xticks(np.round(xe, 2))
    ax.set_yticks(np.round(ze, 2))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("← 왼쪽       포수 시점 (ft)       오른쪽 →", fontsize=8)
    return im


def _footer(fig, text):
    fig.text(0.5, 0.025, text, color="#cfe0f0", fontsize=7.5, ha="center", va="center", zorder=10)


def _fmt_rate(v: float) -> str:
    return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"


# ------------------------------------------------------------ 타자
def plot_batter(df, name: str, team_name: str, summary: Dict, out_path,
                grid: int = C.DEFAULT_GRID, min_ab_zone: int = C.MIN_AB_ZONE,
                demo: bool = False, dpi: int = 130) -> None:
    rate, cnt = M.batter_zone_grid(df, name, grid, min_ab_zone)
    fig = plt.figure(figsize=(6.6, 8.4), dpi=dpi)
    sample = "" if summary["sample_ok"] else " (표본 부족)"
    stadium.add_background(
        fig, title=f"{team_name}  {name}",
        subtitle=(f"타율 {_fmt_rate(summary['avg'])} · 장타율 {_fmt_rate(summary['slg'])} · "
                  f"ISO {_fmt_rate(summary['iso'])} · {summary['type']}{sample}"),
        demo=demo, ylim=(-14.0, 205.0))
    ax = fig.add_axes([0.13, 0.20, 0.70, 0.60], zorder=5)
    ax.set_facecolor(PANEL)
    cax = fig.add_axes([0.855, 0.20, 0.022, 0.60], zorder=5)
    im = _draw_heat(ax, rate, cnt, "RdYlBu_r", 0.0, 0.5, "{:.3f}",
                    highlight=summary["strong_cell"], grid=grid)
    ax.set_ylabel("높이 (ft)", fontsize=9)
    ax.set_title("존별 타율 (타석을 끝낸 공의 위치 기준)", fontsize=10, color="white", fontweight="bold", pad=8)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("타율", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.115,
             f"강한 존  {summary['strong_zone']}     약한 존  {summary['weak_zone']}",
             color="white", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=10)
    _footer(fig, f"타수 {summary['ab']} · 안타 {summary['hits']} · 홈런 {summary['hr']}   |   "
                 f"금색 = 강한 칸(타수 {C.MIN_AB_BEST}↑ 중 최고) · 회색 = 타수 {min_ab_zone} 미만")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------ 투수
def plot_pitcher(df, name: str, team_name: str, summary: Dict, out_path,
                 grid: int = C.DEFAULT_GRID, min_pitch: int = C.MIN_PITCH_ZONE,
                 demo: bool = False, dpi: int = 130) -> None:
    fig = plt.figure(figsize=(11.0, 6.9), dpi=dpi)
    stadium.add_background(
        fig, title=f"{team_name}  {name}",
        subtitle=(f"총 {summary['pitches']}구 · 직구 {summary['fb_share']:.0%} "
                  f"(스트라이크 {summary['fb_strike']:.0%}) · 변화구 {summary['br_share']:.0%} "
                  f"(스트라이크 {summary['br_strike']:.0%})"),
        demo=demo, ylim=(-14.0, 200.0))
    spec = (("직구", "fb", [0.07, 0.17, 0.36, 0.63]), ("변화구", "br", [0.55, 0.17, 0.36, 0.63]))
    last_im = None
    for ptype, key, rect in spec:
        ax = fig.add_axes(rect, zorder=5)
        ax.set_facecolor(PANEL)
        rate, cnt, n = M.pitcher_zone_grid(df, name, ptype, grid, min_pitch)
        last_im = _draw_heat(ax, rate, cnt, "YlGnBu", 0.0, 1.0, "{:.0%}",
                             highlight=summary[f"{key}_best_cell"], grid=grid)
        ax.set_title(f"{ptype} 존별 스트라이크 비율 ({n}구)", fontsize=10, color="white", fontweight="bold", pad=8)
        if key == "fb":
            ax.set_ylabel("높이 (ft)", fontsize=9)
    cax = fig.add_axes([0.94, 0.17, 0.014, 0.63], zorder=5)
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("스트라이크 비율", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.095,
             f"직구 가장 잘 넣는 존  {summary['fb_best_zone']}        "
             f"변화구 가장 잘 넣는 존  {summary['br_best_zone']}",
             color="white", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=10)
    _footer(fig, f"스트라이크 = 루킹·헛스윙·파울 포함   |   금색 = 최고 칸(투구 {C.MIN_PITCH_BEST}↑ 중) · "
                 f"회색 = 투구 {min_pitch}개 미만")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
