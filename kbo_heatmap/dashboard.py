"""분석 결과를 한 장의 HTML 대시보드(잠실의 밤 테마)로 묶습니다. 외부 CDN 없이 동작합니다."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from . import config as C

CSS = """
:root{--bg:#06121f;--panel:#0d2238;--line:#1f3a57;--text:#e6eef7;--muted:#9fb4c9;--gold:#ffd166;--red:#ff7b7b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
 font-family:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR","Noto Sans CJK KR",sans-serif;line-height:1.55}
header img{width:100%;display:block;max-height:340px;object-fit:cover}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h2{margin:40px 0 12px;color:var(--gold);font-size:1.25rem;border-left:4px solid var(--gold);padding-left:10px}
h3{margin:28px 0 10px;font-size:1.05rem}
.banner{background:#3a1d1d;border:1px solid var(--red);color:#ffd0d0;border-radius:8px;padding:10px 14px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:1.5rem;color:var(--gold)}
.card span{color:var(--muted);font-size:.85rem}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--panel)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}
th{background:#112b47;color:var(--gold);position:sticky;top:0}
td.l,th.l{text-align:left}
tr:hover td{background:#12304f}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:.78rem;font-weight:bold}
.pw{background:#5a1f2a;color:#ffc2cb}.ct{background:#1d4a3a;color:#bff0d9}.low{background:#444;color:#ddd}
.warn{color:var(--red);cursor:help}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
figure img{width:100%;display:block}
figcaption{padding:8px 12px;color:var(--muted);font-size:.85rem}
a{color:#8fc7ff}
.note{color:var(--muted);font-size:.85rem}
footer{color:var(--muted);font-size:.8rem;text-align:center;padding:24px 16px;border-top:1px solid var(--line)}
"""


def _e(v) -> str:
    return html.escape(str(v))


def _pct(v, nd=0) -> str:
    return "-" if v != v else f"{v * 100:.{nd}f}%"


def _r3(v) -> str:
    return "-" if v != v else (f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}")


def _name_cell(player) -> str:
    mark = ""
    if not player.verified:
        mark = f' <span class="warn" title="{_e(player.note or "2026 기록 미확인")}">⚠</span>'
    return f"{_e(player.name)}{mark}"


def _type_tag(s: Dict) -> str:
    cls = "pw" if s["type"] == "장타형" else "ct"
    low = '' if s["sample_ok"] else ' <span class="tag low">표본 부족</span>'
    return f'<span class="tag {cls}">{_e(s["type"])}</span>{low}'


def build_dashboard(out_dir: Path, results: List[Dict], meta: Dict) -> Path:
    """results: [{team_code, team_name, role, player, summary, fig}] 목록."""
    batters = [r for r in results if r["role"] == "batter"]
    pitchers = [r for r in results if r["role"] == "pitcher"]
    demo = meta.get("demo", False)

    b_rows = []
    for r in batters:
        s, p = r["summary"], r["player"]
        b_rows.append(
            f'<tr><td class="l">{_e(r["team_name"])}</td><td class="l">{_name_cell(p)}</td>'
            f'<td>{s["ab"]}</td><td>{_r3(s["avg"])}</td><td>{_r3(s["slg"])}</td><td>{_r3(s["iso"])}</td>'
            f'<td>{_type_tag(s)}</td><td>{_e(s["strong_zone"])}</td><td>{_e(s["weak_zone"])}</td>'
            f'<td>{_r3(p.ref.get("avg", float("nan")))}</td><td>{_r3(p.ref.get("ops", float("nan")))}</td>'
            f'<td><a href="figures/{_e(r["fig"])}">그림</a></td></tr>')
    p_rows = []
    for r in pitchers:
        s, p = r["summary"], r["player"]
        era = p.ref.get("era")
        p_rows.append(
            f'<tr><td class="l">{_e(r["team_name"])}</td><td class="l">{_name_cell(p)}</td>'
            f'<td>{s["pitches"]}</td><td>{_pct(s["fb_share"])}</td>'
            f'<td>{_pct(s["fb_strike"])}</td><td>{_pct(s["br_strike"])}</td>'
            f'<td>{_e(s["fb_best_zone"])}</td><td>{_e(s["br_best_zone"])}</td>'
            f'<td>{"-" if era is None else f"{era:.2f}"}</td>'
            f'<td><a href="figures/{_e(r["fig"])}">그림</a></td></tr>')

    teams_order: List[str] = []
    by_team: Dict[str, List[Dict]] = {}
    for r in results:
        if r["team_code"] not in by_team:
            teams_order.append(r["team_code"])
            by_team[r["team_code"]] = []
        by_team[r["team_code"]].append(r)
    team_sections = []
    for code in teams_order:
        figs = "".join(
            f'<figure><a href="figures/{_e(r["fig"])}"><img loading="lazy" src="figures/{_e(r["fig"])}" '
            f'alt="{_e(r["player"].name)} 히트맵"></a><figcaption>'
            f'{"타자" if r["role"] == "batter" else "투수"} · {_name_cell(r["player"])}'
            f'</figcaption></figure>' for r in by_team[code])
        team_sections.append(f'<h3>{_e(by_team[code][0]["team_name"])}</h3><div class="figs">{figs}</div>')

    skipped = meta.get("skipped", [])
    skipped_html = ""
    if skipped:
        items = ", ".join(_e(s) for s in skipped)
        skipped_html = f'<p class="note">데이터가 없어 제외된 선수: {items}</p>'

    banner = ('<div class="banner"><b>DEMO DATA</b> — 이 화면의 모든 수치와 히트맵은 가짜 데이터로 만든 '
              '예시입니다. 실제 선수의 기록이 아닙니다. (실제 투구 데이터는 README 참고)</div>' if demo else "")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KBO 대표 선수 히트맵 · {_e(C.STADIUM["name"])}</title>
<style>{CSS}</style></head>
<body>
<header><img src="field.png" alt="잠실야구장을 형상화한 배경"></header>
<main>
{banner}
<div class="cards">
 <div class="card"><b>{len(batters)}</b><span>분석한 타자</span></div>
 <div class="card"><b>{len(pitchers)}</b><span>분석한 투수</span></div>
 <div class="card"><b>{meta.get("grid")}×{meta.get("grid")}</b><span>히트맵 분할</span></div>
 <div class="card"><b>ISO {meta.get("iso_threshold"):.3f}</b><span>장타형 기준(이상)</span></div>
</div>
{skipped_html}

<h2>타자 요약</h2>
<div class="wrap"><table>
<tr><th class="l">구단</th><th class="l">선수</th><th>타수</th><th>타율</th><th>장타율</th><th>ISO</th>
<th>유형</th><th>강한 존</th><th>약한 존</th><th>시즌 타율<br>(참고)</th><th>시즌 OPS<br>(참고)</th><th></th></tr>
{"".join(b_rows)}</table></div>
<p class="note">강한/약한 존은 타수 {C.MIN_AB_BEST} 이상인 칸 중에서만 고릅니다(포수 시점 기준 좌우).
&quot;시즌 타율/OPS(참고)&quot;는 {_e(meta.get("snapshot", ""))}의 공개 기록이며 위 계산과는 별개입니다.
⚠ 표시는 2026 기록을 확인하지 못한 선수입니다.</p>

<h2>투수 요약</h2>
<div class="wrap"><table>
<tr><th class="l">구단</th><th class="l">선수</th><th>투구 수</th><th>직구 비율</th><th>직구<br>스트라이크</th>
<th>변화구<br>스트라이크</th><th>직구 잘 넣는 존</th><th>변화구 잘 넣는 존</th><th>시즌 ERA<br>(참고)</th><th></th></tr>
{"".join(p_rows)}</table></div>

<h2>구단별 히트맵</h2>
{"".join(team_sections)}
</main>
<footer>생성 {now} · 잠실야구장 배경은 필드 기하 구조를 형상화한 일러스트이며 공식 이미지가 아닙니다.
KBO와 무관한 비공식 학습용 프로젝트입니다.</footer>
</body></html>"""
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path
