"""스트라이크/볼 분류 모델.

문제 정의
  - 정답(라벨): is_strike  (1 = 스트라이크, 0 = 볼)
  - 입력(특징): 공이 홈플레이트를 지날 때의 위치(plate_x, plate_z)와 구종(직구 여부)
  - 목표: 처음 보는 투구의 위치와 구종만 보고 스트라이크인지 볼인지 맞히기

여러 모델을 같은 데이터로 학습해 비교합니다. '기준선'(항상 다수 범주를 찍는 모델)보다 얼마나
나은지를 함께 보여 주기 때문에, 정확도만 높게 나오는 착시를 피할 수 있습니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C

MIN_ROWS = 200               # 이보다 적으면 학습/평가가 의미 없어 오류로 처리
MIN_CLASS_ROWS = 20          # 스트라이크/볼 각각 최소 개수
MAX_ROWS_DEFAULT = 100_000   # 너무 크면 무작위로 이만큼만 사용(실행 시간 보호)
ZONE_CENTER_Z = (C.STRIKE_ZONE[2] + C.STRIKE_ZONE[3]) / 2
ZONE_HALF_W = (C.STRIKE_ZONE[1] - C.STRIKE_ZONE[0]) / 2
ZONE_HALF_H = (C.STRIKE_ZONE[3] - C.STRIKE_ZONE[2]) / 2

RAW = ["plate_x", "plate_z", "is_fast"]
ENGINEERED = RAW + ["abs_x", "z_dist", "zone_dist"]
BASELINE = "기준선(항상 다수 범주)"
FEATURE_DESC = {
    "plate_x": "좌우 위치(ft)", "plate_z": "높이(ft)", "is_fast": "직구 여부(1=직구)",
    "abs_x": "중앙에서 좌우로 떨어진 거리", "z_dist": "존 중심 높이에서 떨어진 거리",
    "zone_dist": "존 중심에서의 정규화 거리",
}


def make_features(x, z, is_fast) -> pd.DataFrame:
    """위치와 구종으로 모델 입력표를 만듭니다(학습과 확률 지도 그리기에 같은 함수를 씁니다)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    f = np.asarray(is_fast, dtype=float) * np.ones_like(x)
    return pd.DataFrame({
        "plate_x": x, "plate_z": z, "is_fast": f,
        "abs_x": np.abs(x),
        "z_dist": np.abs(z - ZONE_CENTER_Z),
        "zone_dist": np.sqrt((x / ZONE_HALF_W) ** 2 + ((z - ZONE_CENTER_Z) / ZONE_HALF_H) ** 2),
    })


def _model_specs(seed: int, rf_trees: int) -> Dict[str, Tuple[Any, List[str]]]:
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  # noqa: E731
    return {
        BASELINE: (DummyClassifier(strategy="most_frequent"), RAW),
        "로지스틱 회귀(원본 좌표)": (lr(), RAW),
        "로지스틱 회귀(파생변수)": (lr(), ENGINEERED),
        "결정트리": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed), RAW),
        "랜덤포레스트": (RandomForestClassifier(n_estimators=rf_trees, min_samples_leaf=3,
                                              random_state=seed, n_jobs=1), RAW),
    }


@dataclass
class ClassificationResult:
    metrics: pd.DataFrame                       # 모델별 평가표 (테스트 데이터 기준)
    models: Dict[str, Any]
    features: Dict[str, List[str]]
    best_name: str
    confusion: np.ndarray                       # 최고 모델 [[TN, FP], [FN, TP]]
    roc: Dict[str, Tuple[np.ndarray, np.ndarray, float]]
    importance: Optional[pd.Series]             # 랜덤포레스트 변수 중요도
    n_rows: int
    n_train: int
    n_test: int
    strike_rate: float                          # 전체에서 스트라이크 비율
    baseline_accuracy: float
    sampled: bool
    seed: int


def prepare_xy(df: pd.DataFrame, max_rows: int = MAX_ROWS_DEFAULT, seed: int = 42):
    """투구표를 (입력표 X, 정답 y, 표본추출 여부)로 바꿉니다. 문제가 있으면 이유와 함께 ValueError."""
    need = ["plate_x", "plate_z", "pitch_type", "is_strike"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"분류에 필요한 컬럼이 없습니다: {missing}")
    d = df.dropna(subset=need)
    d = d[d["is_strike"].isin([0, 1])]
    if len(d) < MIN_ROWS:
        raise ValueError(f"분류를 하기에 데이터가 너무 적습니다({len(d)}행, 최소 {MIN_ROWS}행 필요).")
    counts = d["is_strike"].value_counts()
    if len(counts) < 2:
        raise ValueError("정답이 한 가지(전부 스트라이크 또는 전부 볼)뿐이라 분류할 수 없습니다.")
    if counts.min() < MIN_CLASS_ROWS:
        raise ValueError(f"스트라이크/볼 중 한쪽이 {int(counts.min())}개뿐이라 학습할 수 없습니다"
                         f"(각각 최소 {MIN_CLASS_ROWS}개 필요).")
    sampled = len(d) > max_rows
    if sampled:
        d = d.sample(max_rows, random_state=seed)
    X = make_features(d["plate_x"], d["plate_z"], (d["pitch_type"] == "직구").astype(int))
    y = d["is_strike"].astype(int).to_numpy()
    return X, y, sampled


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, cv: int = 5,
                       max_rows: int = MAX_ROWS_DEFAULT, rf_trees: int = 120) -> ClassificationResult:
    """학습/테스트를 나눠 여러 모델을 비교합니다. 평가는 모두 '처음 보는' 테스트 데이터로 합니다."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size는 0.05~0.5 사이여야 합니다.")
    X, y, sampled = prepare_xy(df, max_rows, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    models, feats, rows, roc = {}, {}, {}, {}
    for name, (model, cols) in _model_specs(seed, rf_trees).items():
        model.fit(X_tr[cols], y_tr)
        pred = model.predict(X_te[cols])
        proba = model.predict_proba(X_te[cols])[:, list(model.classes_).index(1)]
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc = float(roc_auc_score(y_te, proba))
        row = {
            "정확도": accuracy_score(y_te, pred),
            "정밀도": precision_score(y_te, pred, zero_division=0),
            "재현율": recall_score(y_te, pred, zero_division=0),
            "F1": f1_score(y_te, pred, zero_division=0),
            "ROC-AUC": auc,
        }
        if cv and cv >= 2:
            scores = cross_val_score(model, X_tr[cols], y_tr, scoring="accuracy", n_jobs=1,
                                     cv=StratifiedKFold(cv, shuffle=True, random_state=seed))
            row["CV정확도(평균)"], row["CV정확도(표준편차)"] = scores.mean(), scores.std()
        models[name], feats[name], rows[name], roc[name] = model, cols, row, (fpr, tpr, auc)

    metrics = pd.DataFrame(rows).T
    contenders = metrics.drop(index=BASELINE).sort_values(["ROC-AUC", "F1"], ascending=False)
    best = contenders.index[0]
    cm = confusion_matrix(y_te, models[best].predict(X_te[feats[best]]), labels=[0, 1])
    rf = models["랜덤포레스트"]
    importance = pd.Series(rf.feature_importances_, index=feats["랜덤포레스트"]).sort_values(ascending=False)
    return ClassificationResult(
        metrics=metrics, models=models, features=feats, best_name=best, confusion=cm, roc=roc,
        importance=importance, n_rows=len(y), n_train=len(y_tr), n_test=len(y_te),
        strike_rate=float(y.mean()), baseline_accuracy=float(metrics.loc[BASELINE, "정확도"]),
        sampled=sampled, seed=seed)


def probability_grid(result: ClassificationResult, pitch_type: str = "직구", n: int = 60,
                     model_name: Optional[str] = None):
    """위치마다의 '스트라이크 확률' 지도를 만듭니다. (xs, zs, 확률 행렬[z, x])"""
    if pitch_type not in C.PITCH_TYPES:
        raise ValueError(f"pitch_type은 {C.PITCH_TYPES} 중 하나여야 합니다.")
    name = model_name or result.best_name
    model, cols = result.models[name], result.features[name]
    xs = np.linspace(*C.X_RANGE, n)
    zs = np.linspace(*C.Z_RANGE, n)
    xx, zz = np.meshgrid(xs, zs)
    X = make_features(xx.ravel(), zz.ravel(), 1 if pitch_type == "직구" else 0)[cols]
    p = model.predict_proba(X)[:, list(model.classes_).index(1)].reshape(n, n)
    return xs, zs, p


def build_report(result: ClassificationResult, demo: bool = False) -> str:
    """결과를 한글 마크다운 보고서로 정리합니다(숫자만 근거로 문장을 만듭니다)."""
    m, best = result.metrics, result.best_name
    b = m.loc[best]
    gain = b["정확도"] - result.baseline_accuracy
    raw_lr, eng_lr = m.loc["로지스틱 회귀(원본 좌표)"], m.loc["로지스틱 회귀(파생변수)"]
    tn, fp, fn, tp = result.confusion.ravel()
    lines = []
    if demo:
        lines += ["> ⚠ **DEMO DATA**: 이 보고서의 모든 수치는 프로그램이 무작위로 만든 가짜 투구 데이터로 계산한 것입니다. "
                  "제출용 분석에는 실제 투구 데이터를 사용하세요.", ""]
    lines += [
        "# 스트라이크/볼 분류 결과 보고서", "",
        "## 1. 문제 정의",
        "- **분류 목표**: 투구가 홈플레이트를 지날 때의 위치와 구종만 보고 **스트라이크(1) / 볼(0)** 을 맞힌다.",
        "- **정답(라벨)**: `is_strike` (스트라이크 판정이면 1, 볼이면 0)",
        "- **입력(특징)**: " + ", ".join(f"`{k}`({v})" for k, v in FEATURE_DESC.items()),
        "- 분류 종류: 범주가 두 개이므로 **이진 분류**", "",
        "## 2. 데이터와 방법",
        f"- 사용한 투구 수: {result.n_rows:,}개"
        + (f" (원래 데이터가 커서 무작위 표본 {result.n_rows:,}개만 사용)" if result.sampled else ""),
        f"- 스트라이크 비율: {result.strike_rate:.1%} / 볼 비율: {1 - result.strike_rate:.1%}",
        f"- 학습 {result.n_train:,}개 · 테스트 {result.n_test:,}개 (층화 추출, 난수 시드 {result.seed})",
        "- 모든 성능은 **학습에 쓰지 않은 테스트 데이터**로 계산했다.", "",
        "## 3. 모델 비교 (테스트 데이터)", "",
        _md_table(m), "",
        "- **기준선**은 아무 학습 없이 항상 더 많은 범주만 찍는 모델이다. 정확도가 기준선보다 높아야 의미가 있다.",
        "- 정밀도/재현율/F1은 '스트라이크'를 양성으로 계산했다.", "",
        "## 4. 결과 해석",
        f"- 가장 좋은 모델은 **{best}** (ROC-AUC {b['ROC-AUC']:.3f}, 정확도 {b['정확도']:.3f}, F1 {b['F1']:.3f}).",
        f"- 기준선 정확도는 {result.baseline_accuracy:.3f}이고, 최고 모델은 이보다 {gain:+.3f}만큼 "
        + ("높다." if gain > 0 else "높지 않다. 이 경우 모델이 의미 있게 학습했다고 보기 어렵다."),
        f"- 로지스틱 회귀의 정확도는 원본 좌표만 쓸 때 {raw_lr['정확도']:.3f}, 파생변수(중앙에서의 거리 등)를 더하면 "
        f"{eng_lr['정확도']:.3f}이다. 스트라이크존은 사각형에 가까워서 직선 하나로 나누는 모델은 거리 같은 변수를 더해야 잘 맞는다.",
        f"- 최고 모델의 혼동행렬: 볼을 볼로 {tn:,}개, 볼을 스트라이크로 {fp:,}개, "
        f"스트라이크를 볼로 {fn:,}개, 스트라이크를 스트라이크로 {tp:,}개 맞혔다.",
        "- 랜덤포레스트 변수 중요도: "
        + ", ".join(f"`{k}` {v:.2f}" for k, v in result.importance.items()) + ".", "",
        "## 5. 한계",
        "- 위치와 구종만 사용했다. 타자의 스윙 여부, 볼카운트, 타자 키(존 크기), 심판 성향 등은 반영하지 않았다.",
        "- `is_strike`에는 루킹·헛스윙·파울이 모두 들어 있어서, 타자가 휘두른 공은 위치만으로 판정하기 어렵다.",
        "- 같은 투수의 투구가 학습과 테스트에 섞여 있어, 처음 보는 투수에게도 같은 성능이 나온다고 보장할 수 없다.",
        "- 한 번의 무작위 분할 결과이며, 교차검증 표준편차가 클수록 결과가 불안정하다는 뜻이다.",
    ]
    return "\n".join(lines) + "\n"


def _md_table(df: pd.DataFrame) -> str:
    cols = ["모델"] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, row in df.iterrows():
        out.append("| " + " | ".join([str(name)] + [f"{v:.3f}" for v in row]) + " |")
    return "\n".join(out)
