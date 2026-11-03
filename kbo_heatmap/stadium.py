"""잠실야구장을 형상화한 배경 그리기(야간 경기 분위기).

사진이나 로고 같은 저작물은 쓰지 않고, 필드 기하 구조(좌우 100m, 중앙 125m 근사)와
색만으로 matplotlib에서 직접 그립니다. 좌표 단위는 미터이고 홈플레이트가 (0, 0),
+y가 중견수 방향, +x가 1루 쪽입니다.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from . import config as C

# 팔레트
SKY_TOP, SKY_MID, GRASS_EDGE = "#050d18", "#0d2b3e", "#14452a"
GRASS_A, GRASS_B = "#2f8040", "#378a48"
DIRT, DIRT_DARK = "#b9784a", "#8d5a3a"
SEAT, GOLD = "#1c3560", "#ffd166"


def fence_radius(t, left: float, center: float, right: float):
    """t=0(좌측 파울폴) → 0.5(중앙) → 1(우측 파울폴) 일 때 펜스까지의 거리."""
    base = left + (right - left) * np.asarray(t)
    return base + (center - base) * np.sin(np.pi * np.asarray(t))


def fence_points(left: float, center: float, right: float, offset: float = 0.0, n: int = 121):
    t = np.linspace(0.0, 1.0, n)
    r = fence_radius(t, left, center, right) + offset
    theta = np.deg2rad(135.0 - 90.0 * t)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def draw_field(ax, left: Optional[float] = None, center: Optional[float] = None,
               right: Optional[float] = None, alpha: float = 1.0,
               labels: bool = True, lights: bool = True) -> None:
    """ax 위에 잠실야구장(을 닮은) 필드를 그립니다."""
    left = C.STADIUM["left_m"] if left is None else left
    center = C.STADIUM["center_m"] if center is None else center
    right = C.STADIUM["right_m"] if right is None else right
    fp = lambda off: fence_points(left, center, right, offset=off)  # noqa: E731

    # 관중석 / 워닝트랙 / 잔디
    outer, inner = fp(24.0), fp(0.0)
    ax.add_patch(Polygon(np.vstack([outer, inner[::-1]]), closed=True, fc=SEAT, ec="none",
                         alpha=alpha, zorder=1))
    track_o, track_i = fp(0.0), fp(-5.0)
    ax.add_patch(Polygon(np.vstack([track_o, track_i[::-1]]), closed=True, fc=DIRT_DARK,
                         ec="none", alpha=alpha, zorder=2))
    fan = Polygon(np.vstack([[0.0, 0.0], fp(-5.0)]), closed=True, fc=GRASS_A, ec="none",
                  alpha=alpha, zorder=3)
    ax.add_patch(fan)

    # 잔디 줄무늬(홈에서 퍼지는 띠), 팬 모양으로 클리핑
    for i, r in enumerate(np.arange(0.0, center + 10.0, 8.0)):
        stripe = Wedge((0, 0), r + 8.0, 45, 135, width=8.0,
                       fc=GRASS_B if i % 2 else GRASS_A, ec="none", alpha=alpha, zorder=3.1)
        ax.add_patch(stripe)
        stripe.set_clip_path(fan)

    # 내야 흙, 내야 잔디, 홈 주변 흙
    dirt = Circle((0, 18.44), 28.9, fc=DIRT, ec="none", alpha=alpha, zorder=3.2)
    ax.add_patch(dirt)
    dirt.set_clip_path(fan)
    ax.add_patch(Circle((0, 0), 8.0, fc=DIRT, ec="none", alpha=alpha, zorder=3.2))

    b = 27.431 / np.sqrt(2.0)                      # 90ft 베이스 간격
    bases = np.array([[0, 0], [b, b], [0, 2 * b], [-b, b]])
    centroid = bases.mean(axis=0)
    ax.add_patch(Polygon(centroid + (bases - centroid) * 0.72, closed=True, fc=GRASS_A,
                         ec="none", alpha=alpha, zorder=3.3))
    ax.plot(*np.vstack([bases, bases[0]]).T, color="white", lw=0.9, alpha=alpha, zorder=3.4)

    # 파울라인
    for end in (inner[0] * 0.995, inner[-1] * 0.995):
        ax.plot([0, end[0]], [0, end[1]], color="white", lw=1.0, alpha=alpha, zorder=3.4)

    # 마운드, 베이스, 홈플레이트
    ax.add_patch(Circle((0, 18.44), 2.6, fc=DIRT_DARK, ec="none", alpha=alpha, zorder=3.5))
    for bx, by in bases[1:]:
        ax.add_patch(Rectangle((bx - 0.9, by - 0.9), 1.8, 1.8, angle=45,
                               rotation_point="center", fc="white", ec="none",
                               alpha=alpha, zorder=3.6))
    ax.add_patch(Polygon([[-1.0, 0.4], [1.0, 0.4], [1.0, -0.6], [0, -1.6], [-1.0, -0.6]],
                         closed=True, fc="white", ec="none", alpha=alpha, zorder=3.6))

    # 펜스
    ax.plot(*fp(0.0).T, color="#e8eef5", lw=2.0, alpha=alpha, zorder=4)

    # 야간 조명
    if lights:
        pts = fp(34.0)[::8]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, c=GOLD, alpha=0.40 * alpha, zorder=4.5, linewidths=0)

    # 거리 표시
    if labels:
        kw = dict(color="white", fontsize=8, ha="center", va="center", alpha=min(1.0, alpha + 0.2),
                  zorder=5, fontweight="bold")
        ln, cn, rn = fp(-11.0)[0], fp(-11.0)[60], fp(-11.0)[-1]
        ax.text(ln[0] + 6, ln[1] - 4, f"{left:g}m", **kw)
        ax.text(cn[0], cn[1], f"{center:g}m", **kw)
        ax.text(rn[0] - 6, rn[1] - 4, f"{right:g}m", **kw)


def add_background(fig, title: str = "", subtitle: str = "", demo: bool = False,
                   ylim: Tuple[float, float] = (-14.0, 205.0), center_x: float = 0.0,
                   header: bool = True) -> None:
    """figure 전체에 '잠실의 밤' 배경(그라데이션 + 필드 + 헤더)을 깝니다.

    이 함수를 먼저 호출한 뒤 그 위에 zorder>=5인 axes를 얹어 그래프를 그리세요.
    """
    w, h = fig.get_size_inches()
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.axis("off")
    cmap = LinearSegmentedColormap.from_list("night", [GRASS_EDGE, SKY_MID, SKY_TOP])
    bg.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto", cmap=cmap,
              extent=[0, 1, 0, 1], origin="lower")

    fa = fig.add_axes([0, 0, 1, 1], zorder=1)
    fa.axis("off")
    span_y = ylim[1] - ylim[0]
    span_x = span_y * w / h
    fa.set_xlim(center_x - span_x / 2, center_x + span_x / 2)
    fa.set_ylim(*ylim)
    fa.set_aspect("equal", adjustable="box")
    draw_field(fa, alpha=0.55)

    if header:
        fig.text(0.5, 0.968, C.STADIUM["name_en"], color=GOLD, fontsize=9.5, fontweight="bold",
                 ha="center", va="center", zorder=10)
        if title:
            fig.text(0.5, 0.932, title, color="white", fontsize=15, fontweight="bold",
                     ha="center", va="center", zorder=10)
        if subtitle:
            fig.text(0.5, 0.902, subtitle, color="#cfe0f0", fontsize=10,
                     ha="center", va="center", zorder=10)
    if demo:
        fig.text(0.5, 0.5, "DEMO DATA", color="#ff5a5a", fontsize=40, fontweight="bold",
                 alpha=0.18, rotation=28, ha="center", va="center", zorder=9)


def save_hero(path, title: str, subtitle: str, demo: bool = False) -> None:
    """대시보드 상단 배너 이미지를 저장합니다."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.2), dpi=110)
    add_background(fig, header=False, ylim=(-8.0, 165.0), center_x=-95.0)
    fig.text(0.04, 0.80, C.STADIUM["name_en"], color=GOLD, fontsize=13, fontweight="bold",
             ha="left", va="center", zorder=10)
    fig.text(0.04, 0.62, title, color="white", fontsize=26, fontweight="bold",
             ha="left", va="center", zorder=10)
    fig.text(0.04, 0.46, subtitle, color="#cfe0f0", fontsize=13, ha="left", va="center", zorder=10)
    if demo:
        fig.text(0.04, 0.30, "DEMO DATA — 가짜 데이터로 만든 예시입니다", color="#ff8a8a",
                 fontsize=12, fontweight="bold", ha="left", va="center", zorder=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)
