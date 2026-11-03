# 데이터 출처와 주의사항

이 프로젝트의 가장 큰 난관은 **투구 단위 위치 데이터(plate_x, plate_z)** 입니다. 타율·장타율은 기록실 수치로도
계산되지만, 존별 히트맵은 투구마다의 좌표가 있어야 그릴 수 있습니다.

## 확인된 사실 (2026-10 기준)

- **KBO 공식 기록실(koreabaseball.com)** 은 자동 접근(robots)을 허용하지 않는 것으로 확인되었습니다.
  이 프로젝트는 크롤러를 포함하지 않습니다. 시즌 기록은 사이트에서 직접 확인하세요.
- **MyKBO Stats(mykbostats.com, 비공식)** 의 순위표로 `data/players.json`의 `ref`(참고 기록)를 채웠습니다
  (2026-10-14 기준). 시즌이 진행·종료되면 값이 바뀔 수 있습니다.
- KBO는 PTS(투구 추적) 등 트래킹 데이터를 쓰고 있으나, Statcast처럼 누구나 내려받을 수 있는 공개 API는
  확인하지 못했습니다.
- GitHub의 `dabinj/KBO_VISION`은 네이버 스포츠 중계 데이터에서 구종·위치·구속 등을 수집한다고 소개되어
  있습니다(검색으로 확인한 소개 기준, 코드를 직접 검증하지는 않았습니다). 투구 위치 데이터를 구하는 방법을
  찾을 때 참고하기 좋습니다.

## 분류 모델용 실제 데이터: MLB Statcast

- Baseball Savant(baseballsavant.mlb.com)의 Statcast Search는 투구별 위치(`plate_x`, `plate_z`)와 판정(`type`: B/S/X)을
  CSV로 내려받게 해 줍니다. 이 파일을 `python -m kbo_heatmap.statcast --from-csv ...`로 변환하면 분류 모델에 바로 쓸 수 있습니다.
- `pybaseball` 라이브러리로도 받을 수 있지만, 이 개발 환경에서는 인터넷이 막혀 있어 **직접 실행해 보지 못했습니다**.
  실패하면 Savant CSV 방식을 쓰세요.
- KBO가 아니라 MLB 데이터임을 보고서에 분명히 적어야 합니다.

## 데이터를 구한 뒤

1. `data/pitches_template.csv`와 같은 컬럼으로 맞춰 CSV를 만듭니다. (`data/README.md` 참고)
2. 선수 이름이 `data/players.json`과 정확히 일치하는지 확인합니다(외국인 선수 표기 주의).
3. `python -m kbo_heatmap --data data/pitches_2026.csv`

## 지켜야 할 것

- 각 사이트의 **이용약관과 robots.txt**를 확인하고, 허용되지 않은 자동 수집은 하지 마세요.
- 요청 간격을 충분히 두고, 한 번 받은 데이터는 CSV로 저장해 재사용하세요.
- 수집한 원본 데이터를 저장소에 그대로 올려 재배포하지 마세요(`.gitignore`에 `data/pitches*.csv`가 포함되어 있습니다).
