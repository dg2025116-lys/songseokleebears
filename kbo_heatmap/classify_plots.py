"""분류 결과 그림: 스트라이크 확률 지도, 평가 한 장(혼동행렬 · ROC · 모델별 F1). 배경은 잠실의 밤."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import classify as K
from . import config as C
from . import stadium
from .plots import PANEL

ZONE_LINE = "#0b1b2e"
OPAQUE = "#f4f7fb"   # 선 그래프는 배경 필드가 비치면 읽기 어려워 불투명하게


def plot_probability_map(result: K.ClassificationResult, out, demo: bool = False, dpi: int = 130) -> None:
    """직구/변화구별 '스트라이크 확률' 지도. 빨간 점선이 모델이 그은 결정 경계(확률 0.5)입니다."""
    fig = plt.figure(figsize=(11.0, 6.9), dpi=dpi)
    stadium.add_background(
        fig, title="분류 모델이 본 스트라이크 확률 지도",
        subtitle=f"모델: {result.best_name}  ·  ROC-AUC {result.metrics.loc[result.best_name, 'ROC-AUC']:.3f}"
                 f"  ·  정확도 {result.metrics.loc[result.best_name, '정확도']:.3f}",
        demo=demo, ylim=(-14.0, 200.0))
    im = None
    for ptype, rect in (("직구", [0.07, 0.17, 0.36, 0.63]), ("변화구", [0.55, 0.17, 0.36, 0.63])):
        ax = fig.add_axes(rect, zorder=5)
        ax.set_facecolor(PANEL)
        xs, zs, p = K.probability_grid(result, ptype)
        im = ax.imshow(p, origin="lower", cmap="YlGnBu", vmin=0, vmax=1, aspect="auto",
                       extent=[*C.X_RANGE, *C.Z_RANGE])
        if p.min() < 0.5 < p.max():
            ax.contour(xs, zs, p, levels=[0.5], colors="#d00000", linewidths=2.2, linestyles="--")
        x0, x1, z0, z1 = C.STRIKE_ZONE
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=ZONE_LINE, lw=2.4, zorder=5))
        ax.set_title(f"{ptype}: 위치별 스트라이크 확률", fontsize=10, color="white", fontweight="bold", pad=8)
        ax.set_xlabel("← 왼쪽       포수 시점 (ft)       오른쪽 →", fontsize=8)
        ax.tick_params(labelsize=7)
        if ptype == "직구":
            ax.set_ylabel("높이 (ft)", fontsize=9)
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.17, 0.014, 0.63], zorder=5))
    cb.set_label("스트라이크 확률", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.095, "진할수록 스트라이크일 가능성이 높음   ·   빨간 점선 = 확률 50% 경계   ·   검은 사각형 = 대략적인 스트라이크존",
             color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=10)
    fig.text(0.5, 0.025, f"학습 {result.n_train:,}개로 학습한 모델이 만든 지도입니다 (실제 판정이 아니라 모델의 예측)",
             color="#cfe0f0", fontsize=7.5, ha="center", va="center", zorder=10)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_evaluation(result: K.ClassificationResult, out, demo: bool = False, dpi: int = 130) -> None:
    """혼동행렬(최고 모델) · ROC 곡선(전체 모델) · 모델별 F1."""
    fig = plt.figure(figsize=(13.0, 6.2), dpi=dpi)
    stadium.add_background(
        fig, title="분류 모델 평가 (처음 보는 테스트 데이터 기준)",
        subtitle=f"테스트 {result.n_test:,}개  ·  스트라이크 비율 {result.strike_rate:.1%}  ·  "
                 f"기준선 정확도 {result.baseline_accuracy:.3f}",
        demo=demo, ylim=(-14.0, 215.0))

    # 1) 혼동행렬
    ax = fig.add_axes([0.10, 0.17, 0.22, 0.58], zorder=5)
    ax.set_facecolor(OPAQUE)
    cm = result.confusion
    ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#102030")
    ax.set_xticks([0, 1], ["볼", "스트라이크"])
    ax.set_yticks([0, 1], ["볼", "스트라이크"])
    ax.set_xlabel("모델의 예측", fontsize=9)
    ax.set_ylabel("실제 정답", fontsize=9)
    ax.set_title(f"혼동행렬 ({result.best_name})", fontsize=10, color="white", fontweight="bold", pad=8)

    # 2) ROC
    ax = fig.add_axes([0.42, 0.17, 0.25, 0.58], zorder=5)
    ax.set_facecolor(OPAQUE)
    for name, (fpr, tpr, auc) in result.roc.items():
        if name == K.BASELINE:
            continue
        ax.plot(fpr, tpr, lw=2.4 if name == result.best_name else 1.4, label=f"{name} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="무작위(0.50)")
    ax.set_xlabel("거짓 양성 비율 (볼인데 스트라이크로 예측)", fontsize=8)
    ax.set_ylabel("참 양성 비율 (스트라이크를 맞힘)", fontsize=8)
    ax.set_title("ROC 곡선 (범례 = AUC)", fontsize=10, color="white", fontweight="bold", pad=8)
    ax.legend(fontsize=6.5, loc="lower right", facecolor="white", framealpha=0.9)
    ax.tick_params(labelsize=7)

    # 3) 모델별 F1
    ax = fig.add_axes([0.77, 0.17, 0.21, 0.58], zorder=5)
    ax.set_facecolor(OPAQUE)
    names = list(result.metrics.index)
    f1 = result.metrics["F1"].to_numpy()
    colors = ["#9aa8b8" if n == K.BASELINE else ("#ffd166" if n == result.best_name else "#3b82c4") for n in names]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, f1, color=colors)
    for y, v in zip(ypos, f1):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(ypos, [n.replace("(", "\n(") for n in names], fontsize=6.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 점수", fontsize=8)
    ax.set_title("모델별 F1 (회색 = 기준선)", fontsize=10, color="white", fontweight="bold", pad=8)
    ax.tick_params(axis="x", labelsize=7)

    fig.text(0.5, 0.06, f"금색 = 최고 모델({result.best_name})  ·  ROC-AUC가 0.5면 무작위와 같고 1.0에 가까울수록 좋음",
             color="white", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=10)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
