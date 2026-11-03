"""선수 설정/투구 데이터 로딩, 검증, 정규화, 데모 데이터 생성."""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("pitcher", "batter", "pitch_type", "is_strike")


# ============================================================ 선수 설정
@dataclass
class Player:
    name: str
    role: str                      # 'batter' | 'pitcher'
    verified: bool = True
    note: str = ""
    ref: Dict[str, float] = field(default_factory=dict)


@dataclass
class Team:
    code: str
    name: str
    batters: List[Player]
    pitchers: List[Player]


def load_players(path: Optional[Path] = None) -> List[Team]:
    """data/players.json 형식의 파일을 읽어 팀 목록으로 반환합니다."""
    path = Path(path) if path else C.DEFAULT_PLAYERS_PATH
    if not path.exists():
        raise FileNotFoundError(f"선수 설정 파일을 찾을 수 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "teams" not in raw or not isinstance(raw["teams"], list):
        raise ValueError("players.json에 'teams' 목록이 없습니다.")

    teams: List[Team] = []
    seen_codes = set()
    seen = {"batter": set(), "pitcher": set()}
    for t in raw["teams"]:
        code, name = str(t.get("code", "")).strip(), str(t.get("name", "")).strip()
        if not code or not name:
            raise ValueError(f"팀의 code/name이 비어 있습니다: {t}")
        if code.upper() in seen_codes:
            raise ValueError(f"팀 코드가 중복되었습니다: {code}")
        seen_codes.add(code.upper())
        groups = {}
        for key, role in (("batters", "batter"), ("pitchers", "pitcher")):
            players = []
            for p in t.get(key, []):
                pname = str(p.get("name", "")).strip()
                if not pname:
                    raise ValueError(f"{code}의 {key}에 이름이 빈 선수가 있습니다.")
                if pname in seen[role]:
                    raise ValueError(
                        f"{role} 이름이 중복되었습니다: {pname} "
                        "(데이터가 이름으로 선수를 구분하므로 중복을 허용하지 않습니다)")
                seen[role].add(pname)
                players.append(Player(
                    name=pname, role=role,
                    verified=bool(p.get("verified", True)),
                    note=str(p.get("note", "")),
                    ref={k: float(v) for k, v in (p.get("ref") or {}).items()},
                ))
            groups[key] = players
        teams.append(Team(code=code, name=name,
                          batters=groups["batters"], pitchers=groups["pitchers"]))
    return teams


# ============================================================ 투구 데이터
def zone_to_xy(code: str):
    """25분할 존 코드(예: 'C3')를 칸 중심 좌표(ft)로 변환합니다.

    가정: 알파벳 A~E = 높은 쪽에서 낮은 쪽, 숫자 1~5 = 포수 시점 왼쪽에서 오른쪽.
    데이터 출처의 존 배치가 다르면 이 함수를 수정하세요.
    """
    code = str(code).strip().upper()
    if len(code) != 2 or code[0] not in "ABCDE" or code[1] not in "12345":
        raise ValueError(f"올바르지 않은 존 코드: {code!r} (A1~E5 형식이어야 합니다)")
    row, col = "ABCDE".index(code[0]), int(code[1]) - 1
    xe = np.linspace(*C.X_RANGE, 6)
    ze = np.linspace(*C.Z_RANGE, 6)[::-1]
    return (xe[col] + xe[col + 1]) / 2, (ze[row] + ze[row + 1]) / 2


def normalize_pitch_type(value) -> Optional[str]:
    """구종 문자열을 '직구' / '변화구'로 통일합니다. 모르는 값은 None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().upper()
    if key in C.FASTBALL_NAMES:
        return "직구"
    if key in C.BREAKING_NAMES:
        return "변화구"
    return None


def prepare_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """투구 데이터프레임을 검증하고 분석용으로 정규화합니다(여러 번 호출해도 안전)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing}. 필요한 컬럼: {list(REQUIRED_COLUMNS)} "
            "+ (plate_x, plate_z) 또는 zone, + pa_result(선택)")
    out = df.copy()

    # --- 좌표
    has_xy = {"plate_x", "plate_z"}.issubset(out.columns)
    if not has_xy:
        if "zone" not in out.columns:
            raise ValueError("위치 정보가 없습니다. plate_x/plate_z 또는 zone(A1~E5) 컬럼이 필요합니다.")
        xy = out["zone"].map(lambda z: zone_to_xy(z) if pd.notna(z) else (np.nan, np.nan))
        out["plate_x"] = [p[0] for p in xy]
        out["plate_z"] = [p[1] for p in xy]
    out["plate_x"] = pd.to_numeric(out["plate_x"], errors="coerce")
    out["plate_z"] = pd.to_numeric(out["plate_z"], errors="coerce")

    # --- 이름/구종
    for col in ("pitcher", "batter"):
        out[col] = out[col].astype("string").str.strip()
    out["pitch_type"] = out["pitch_type"].map(normalize_pitch_type)

    # --- 스트라이크 여부
    col = out["is_strike"]
    if pd.api.types.is_bool_dtype(col):
        out["is_strike"] = col.astype(int)
    elif not pd.api.types.is_numeric_dtype(col):      # 'Y'/'N', 'true'/'false' 등 텍스트(pandas 버전과 무관하게 처리)
        out["is_strike"] = col.map(
            lambda v: np.nan if pd.isna(v) else
            1 if str(v).strip().lower() in ("1", "1.0", "true", "t", "y", "yes", "스트라이크") else
            (0 if str(v).strip().lower() in ("0", "0.0", "false", "f", "n", "no", "볼") else np.nan))
    out["is_strike"] = pd.to_numeric(out["is_strike"], errors="coerce")

    # --- 타석 결과
    if "pa_result" not in out.columns:
        logger.warning("pa_result 컬럼이 없어 타자 분석(타율/장타 유형)을 할 수 없습니다.")
        out["pa_result"] = np.nan
    out["pa_result"] = out["pa_result"].astype("object").map(
        lambda v: np.nan if pd.isna(v) or str(v).strip() == "" else str(v).strip())

    # --- 사용할 수 없는 행 제거(개수는 알려줌)
    n0 = len(out)
    bad_type = out["pitch_type"].isna()
    bad_strike = ~out["is_strike"].isin([0, 1])
    bad_xy = out["plate_x"].isna() | out["plate_z"].isna()
    bad_name = out["pitcher"].isna() | out["batter"].isna()
    for label, mask in (("구종 인식 불가", bad_type), ("is_strike 값 오류", bad_strike),
                        ("위치 결측", bad_xy), ("이름 결측", bad_name)):
        if mask.any():
            logger.warning("%s: %d행 (전체 %d행 중)", label, int(mask.sum()), n0)
    out = out[~(bad_type | bad_strike | bad_xy | bad_name)].copy()
    out["is_strike"] = out["is_strike"].astype(int)
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("검증 후 남은 투구 데이터가 없습니다. 컬럼 형식을 확인하세요.")
    return out


def load_pitches_bytes(raw: bytes) -> pd.DataFrame:
    """CSV 파일 내용(bytes)을 읽어 정규화합니다. UTF-8(BOM 무관)과 CP949(엑셀 한글 저장)를 지원합니다."""
    if not raw:
        raise ValueError("빈 파일입니다.")
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949(엑셀) CSV만 지원합니다.")
    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise ValueError("CSV에 내용이 없습니다.") from None
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV 형식 오류: {e}") from None
    return prepare_pitches(df)


def load_pitches(path) -> pd.DataFrame:
    """CSV 파일 경로를 읽어 정규화합니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"투구 데이터 파일을 찾을 수 없습니다: {path}")
    return load_pitches_bytes(path.read_bytes())


# ============================================================ 데모 데이터
def make_demo(teams: List[Team], seed: int = 2026, pitches_per_pitcher: int = 1800) -> pd.DataFrame:
    """실제 데이터가 없을 때 전체 흐름을 시험하기 위한 가짜 투구 데이터.

    ⚠ 분석 결과는 의미가 없습니다. 타자의 참고 기록(ref)을 이용해 장타/단타 성향만
    대략 흉내 내고, 선수마다 '강한 존'을 하나씩 심어 둡니다.
    """
    rng = np.random.default_rng(seed)
    batters = [b for t in teams for b in t.batters]
    pitchers = [p for t in teams for p in t.pitchers]
    if not batters or not pitchers:
        raise ValueError("데모 데이터를 만들려면 타자와 투수가 각각 1명 이상 필요합니다.")

    names = np.array([b.name for b in batters])
    prof = []
    for b in batters:
        avg = b.ref.get("avg", 0.270)
        slg = b.ref.get("slg", avg + 0.12)
        prof.append({
            "avg": avg,
            "iso": max(slg - avg, 0.03),
            "hot": (rng.uniform(-0.6, 0.6), rng.uniform(1.8, 3.2)),   # 강한 존 중심
        })

    x0, x1, z0, z1 = C.STRIKE_ZONE
    frames = []
    for p in pitchers:
        n = pitches_per_pitcher
        p_fb = rng.uniform(0.50, 0.65)
        fb_c = (rng.uniform(-0.25, 0.25), rng.uniform(2.5, 2.9))
        br_c = (rng.uniform(-0.35, 0.35), rng.uniform(1.6, 2.1))
        is_fb = rng.random(n) < p_fb
        cx = np.where(is_fb, fb_c[0], br_c[0])
        cz = np.where(is_fb, fb_c[1], br_c[1])
        sd = np.where(is_fb, 0.55, 0.68)
        x = rng.normal(cx, sd)
        z = rng.normal(cz, sd * 1.1)
        in_zone = (x > x0) & (x < x1) & (z > z0) & (z < z1)
        is_strike = (rng.random(n) < np.where(in_zone, 0.86, 0.24)).astype(int)
        bidx = rng.integers(0, len(batters), n)
        ends = rng.random(n) < 0.23

        results = np.full(n, None, dtype=object)
        for i in np.flatnonzero(ends):
            pr = prof[bidx[i]]
            if not in_zone[i] and rng.random() < 0.20:
                results[i] = "볼넷"
                continue
            hx, hz = pr["hot"]
            hot = 1.0 + 0.9 * np.exp(-((x[i] - hx) ** 2 + (z[i] - hz) ** 2) / (2 * 0.35 ** 2))
            p_hit = min(pr["avg"] / 1.15 * (1.15 if in_zone[i] else 0.5) * hot, 0.65)
            if rng.random() < p_hit:
                p_xbh = float(np.clip(pr["iso"] / 0.45, 0.12, 0.65))
                if rng.random() < p_xbh:
                    results[i] = rng.choice(["2루타", "3루타", "홈런"], p=[0.40, 0.05, 0.55])
                else:
                    results[i] = "1루타"
            else:
                results[i] = rng.choice(["아웃", "삼진", "병살"], p=[0.62, 0.33, 0.05])

        frames.append(pd.DataFrame({
            "pitcher": p.name,
            "batter": names[bidx],
            "pitch_type": np.where(is_fb, "직구", "변화구"),
            "plate_x": x, "plate_z": z,
            "is_strike": is_strike,
            "pa_result": results,
        }))
    return prepare_pitches(pd.concat(frames, ignore_index=True))
