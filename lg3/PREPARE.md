# LG3 데이터 준비 (초안)

이 문서는 현재 LG3 데이터 준비 흐름과 최근 수정사항을 정리한 초안입니다.

## TODO 스크립트와의 차이 요약

`lg3/TODO/하루 전력량 에측 전처리.PY` 대비 현재 준비 스크립트의 차이점입니다.

- 시간 단위: TODO는 1시간, 현재는 15분(또는 `--freq`로 설정).
- EREPORT 처리: TODO는 커버리지/결측 보정이 있고, 현재는 단순 리샘플 평균.
- SMARTCARE 집계: TODO는 유닛별 평균 후 평균, 현재는 구간 전체 최빈값.
- 휴일 피처: TODO는 포함, 현재는 미포함(추후 고려).

## 나중에 고려할 항목

- 휴일 정보를 추가하여 `is_holiday` 같은 피처를 넣는 방안을 추후 반영합니다.

## CSV 에러 처리 및 Pdb

CSV 파싱 문제가 있으면 Pdb로 멈춰서 사용자가 확인 후 `c`로 진행합니다.

- 파싱 실패(토크나이징/IO)는 디버그 정보 출력 후 스킵 처리합니다.
- 컬럼 누락 시 사용 가능한 컬럼/alias 정보를 출력합니다.
- Entha 누락은 스킵 전 확인을 요구합니다.

## 데이터 경로

- EREPORT 입력: `data/elec1_f2/EREPORT`
- SMARTCARE 입력: `data/elec1_f2/SMARTCARE`
- 출력: `data/elec1_f2/processed`

## 실행 스크립트

단일 스크립트를 사용합니다.

```bash
bash ./lg3/scripts/prepare_lg3_data.sh
```

스크립트 상단에서 주요 컬럼을 정의합니다.

- `EREPORT_COLS`: EREPORT 컬럼 목록 (콤마 구분)
- `SMARTCARE_COLS`: SMARTCARE 컬럼 목록 (기본은 `Tod`)

## 시간 정렬 규칙

- EREPORT는 left label/left closed로 리샘플링합니다.
- SMARTCARE는 `floor(freq)`로 같은 left label 기준을 맞춥니다.
- 결과적으로 EREPORT와 SMARTCARE가 같은 15분(또는 설정 단위) 타임라인으로 합쳐집니다.
- CSV가 스킵되어 특정 시간대가 비면, 병합 후 `dropna()`에서 해당 시간대가 제거됩니다.

## SMARTCARE 집계 방식

- 같은 시간 구간의 전체 값(모든 `Auto Id` 포함)에서 최빈값을 선택합니다.
- 커버리지 체크는 제거되어 있으며, 충분히 관측되었다고 가정합니다.

## Entha 컬럼 처리

- `VAP_Entha` 또는 `LIQ_Entha`가 없으면 `*_R32` alias를 우선 사용합니다.
- 원본/alias 모두 없으면 Pdb로 확인 후 해당 CSV를 스킵합니다.