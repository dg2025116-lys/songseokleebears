"""스트라이크/볼 분류 실행: python -m kbo_heatmap.classify_cli --demo / --data CSV"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from . import classify as K  # noqa: E402
from . import classify_plots as P  # noqa: E402
from . import plots  # noqa: E402
from .data import load_pitches, load_players, make_demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kbo_heatmap.classify_cli",
        description="투구 위치와 구종으로 스트라이크/볼을 맞히는 분류 모델을 학습·평가합니다.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="가짜 데이터로 전체 흐름을 시험합니다.")
    src.add_argument("--data", metavar="CSV", help="투구 단위 CSV (data/README.md 형식)")
    ap.add_argument("--out", default="output_classification", metavar="DIR", help="결과 폴더")
    ap.add_argument("--test-size", type=float, default=0.2, help="테스트 비율 (기본 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드 (같으면 결과도 같음)")
    ap.add_argument("--cv", type=int, default=5, help="교차검증 폴드 수, 0이면 생략 (기본 5)")
    ap.add_argument("--max-rows", type=int, default=K.MAX_ROWS_DEFAULT, help="사용할 최대 투구 수")
    ap.add_argument("--font-path", metavar="FILE", help="한글 폰트 파일 직접 지정")
    return ap


def run(args: argparse.Namespace) -> int:
    df = make_demo(load_players()) if args.demo else load_pitches(args.data)
    plots.setup_font(args.font_path)
    result = K.train_and_evaluate(df, test_size=args.test_size, seed=args.seed, cv=args.cv,
                                  max_rows=args.max_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.metrics.round(4).to_csv(out / "metrics.csv", encoding="utf-8-sig")
    P.plot_probability_map(result, out / "probability_map.png", demo=args.demo)
    P.plot_evaluation(result, out / "evaluation.png", demo=args.demo)
    (out / "classification_report.md").write_text(K.build_report(result, demo=args.demo), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result.metrics.round(3).to_string())
    print(f"\n최고 모델: {result.best_name}  (기준선 정확도 {result.baseline_accuracy:.3f})")
    if args.demo:
        print("※ DEMO 데이터: 수치는 의미가 없습니다. 제출용 분석에는 실제 투구 데이터를 쓰세요.")
    print(f"결과 저장: {out.resolve()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
