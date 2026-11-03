"""명령줄 실행: python -m kbo_heatmap --demo / --data pitches.csv"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장

import pandas as pd  # noqa: E402

from . import __version__  # noqa: E402
from . import config as C  # noqa: E402
from . import metrics as M  # noqa: E402
from . import plots, stadium  # noqa: E402
from .dashboard import build_dashboard  # noqa: E402
from .tables import batter_row, pitcher_row  # noqa: E402
from .data import load_pitches, load_players, make_demo  # noqa: E402

logger = logging.getLogger("kbo_heatmap")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kbo_heatmap",
        description="KBO 구단 대표 선수의 타율·장타 유형·존별 히트맵을 잠실야구장 테마로 만듭니다.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="가짜 데이터로 전체 흐름을 시험합니다.")
    src.add_argument("--data", metavar="CSV", help="투구 단위 CSV 파일 경로")
    ap.add_argument("--players", metavar="JSON", default=str(C.DEFAULT_PLAYERS_PATH),
                    help="대표 선수 설정 파일 (기본: data/players.json)")
    ap.add_argument("--out", metavar="DIR", default=str(C.DEFAULT_OUT_DIR), help="결과 폴더 (기본: output)")
    ap.add_argument("--teams", metavar="CODES", help="일부 구단만 처리. 예: KIA,LG,DOOSAN")
    ap.add_argument("--grid", type=int, default=C.DEFAULT_GRID, choices=C.SUPPORTED_GRIDS,
                    help="히트맵 분할 수 (기본 5)")
    ap.add_argument("--iso-threshold", type=float, default=C.ISO_THRESHOLD,
                    help=f"ISO가 이 값 이상이면 장타형 (기본 {C.ISO_THRESHOLD})")
    ap.add_argument("--seed", type=int, default=2026, help="데모 데이터 난수 시드")
    ap.add_argument("--font-path", metavar="FILE", help="한글 폰트 파일(.ttf/.otf) 직접 지정")
    ap.add_argument("--no-dashboard", action="store_true", help="HTML 대시보드를 만들지 않습니다.")
    ap.add_argument("-v", "--verbose", action="store_true", help="자세한 로그")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def _select_teams(teams, codes: Optional[str]):
    if not codes:
        return teams
    wanted = [c.strip().upper() for c in codes.split(",") if c.strip()]
    known = {t.code.upper(): t for t in teams}
    unknown = [c for c in wanted if c not in known]
    if unknown:
        raise ValueError(f"알 수 없는 구단 코드: {unknown}. 사용 가능: {sorted(known)}")
    return [known[c] for c in wanted]


def run(args: argparse.Namespace) -> int:
    all_teams = load_players(Path(args.players))
    teams = _select_teams(all_teams, args.teams)

    if args.demo:
        df = make_demo(all_teams, seed=args.seed)   # 선수 전체로 만들어 두고 선택 구단만 그림
        logger.info("데모 데이터 생성: %d구", len(df))
    else:
        df = load_pitches(args.data)
        logger.info("투구 데이터 로드: %d구", len(df))

    font = plots.setup_font(args.font_path)
    logger.info("사용 폰트: %s", font or "기본 폰트(한글 깨짐 가능)")

    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    skipped: List[str] = []
    for team in teams:
        for idx, player in enumerate(team.batters, 1):
            summ = M.batter_summary(df, player.name, args.grid, args.iso_threshold)
            if summ is None:
                logger.warning("[건너뜀] 타자 데이터 없음: %s %s", team.code, player.name)
                skipped.append(f"{team.name} {player.name}(타자)")
                continue
            fname = f"batter_{team.code}_{idx}.png"
            plots.plot_batter(df, player.name, team.name, summ, fig_dir / fname,
                              grid=args.grid, demo=args.demo)
            results.append(dict(team_code=team.code, team_name=team.name, role="batter",
                                player=player, summary=summ, fig=fname))
            logger.info("타자 완료: %s %s", team.code, player.name)
        for idx, player in enumerate(team.pitchers, 1):
            summ = M.pitcher_summary(df, player.name, args.grid)
            if summ is None:
                logger.warning("[건너뜀] 투수 데이터 없음: %s %s", team.code, player.name)
                skipped.append(f"{team.name} {player.name}(투수)")
                continue
            fname = f"pitcher_{team.code}_{idx}.png"
            plots.plot_pitcher(df, player.name, team.name, summ, fig_dir / fname,
                               grid=args.grid, demo=args.demo)
            results.append(dict(team_code=team.code, team_name=team.name, role="pitcher",
                                player=player, summary=summ, fig=fname))
            logger.info("투수 완료: %s %s", team.code, player.name)

    if not results:
        logger.error("처리된 선수가 없습니다. 선수 이름이 데이터의 batter/pitcher 값과 같은지 확인하세요.")
        return 1

    _write_csv(out, results)
    if not args.no_dashboard:
        stadium.save_hero(out / "field.png", "KBO 대표 선수 분석",
                          "타율 · 장타 유형 · 존별 히트맵", demo=args.demo)
        page = build_dashboard(out, results, dict(
            demo=args.demo, grid=args.grid, iso_threshold=args.iso_threshold,
            skipped=skipped, snapshot="MyKBO Stats 2026-10-14"))
        logger.info("대시보드: %s", page)

    nb = sum(r["role"] == "batter" for r in results)
    npit = sum(r["role"] == "pitcher" for r in results)
    print(f"완료: 타자 {nb}명, 투수 {npit}명 → {out.resolve()}")
    if skipped:
        print(f"제외된 선수 {len(skipped)}명: {', '.join(skipped)}")
    return 0


def _write_csv(out: Path, results: List[dict]) -> None:
    brows = [batter_row(r["team_name"], r["player"], r["summary"]) for r in results if r["role"] == "batter"]
    prows = [pitcher_row(r["team_name"], r["player"], r["summary"]) for r in results if r["role"] == "pitcher"]
    pd.DataFrame(brows).to_csv(out / "batter_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prows).to_csv(out / "pitcher_summary.csv", index=False, encoding="utf-8-sig")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
