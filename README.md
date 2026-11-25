# hl — L2 재생 위 호가 정책 실험

가설로 둔 정책 형태: 상태 x와 재고 q에 따라 양쪽 깊이 δ_b(x,q), δ_a(x,q) ∈ [0, ∞]를 고른다(∞ = 철회).
|q|가 q*를 넘으면 테이커로 털고, 비청산 공격 흐름이 한쪽으로 몰리면 그쪽을 취소하고, 청산 흐름이 몰리면 깊은 레벨 d_liq에서 받는다.
고전 MM(x 무시, 터치 근처)과 깊은 받기(d0=∞, liq_on 유한, q_max=1)는 이 형태의 두 구석이다(`hl/run.py` PRESETS).

```
hl/data.py    S3 원본(lz4) → 코인·일별 npz
hl/sim.py     L2 스냅샷 + 체결 이벤트 재생, 정책, 추정기
hl/run.py     워크포워드: 탐침(probe) → 추정기 학습(직전 train일) → 정책 → 요약
hl/stats.py   t, Deflated Sharpe
scripts/fetch.sh   S3 다운로드(요청자 부담)
tests/        합성 검정(큐·지연·관통·ALO·회계·파서·심은 신호): python -m pytest -q
```

## 실행 (VPS, Ubuntu)
```
pip install -r requirements.txt
scripts/fetch.sh BTC 20261001 20261021                  # → data/raw/
python -m hl.data --coin BTC --dates 20261001:20261021   # → data/day/BTC/YYYYMMDD.npz
python -m hl.run --coin BTC --train 5 --jobs 4           # → results/<시각>_BTC/
```
체결 경로(`node_fills_by_block/hourly/<날짜>/`)는 처음 한 번 `aws s3 ls ... --request-payer requester`로 확인한다. fetch.sh가 비어 있으면 확인 명령을 출력한다.

## 데이터 계약
| 단계 | 경로 | 내용 |
|---|---|---|
| 원본 L2 | `data/raw/l2/<COIN>/<날짜>/<H>/l2Book/<COIN>.lz4` | `s3://hyperliquid-archive/market_data/...`, 0.5초 스냅샷, 최대 20레벨 |
| 원본 체결 | `data/raw/fills/<날짜>/<H>` | `s3://hl-mainnet-node-data/node_fills_by_block/...`, 전 코인 |
| 일별 | `data/day/<COIN>/<날짜>.npz` | `t nb na bpx bsz apx asz` / `tt tpx tsz tside(+1 매수 공격) tliq` |
| 결과 | `results/<시각>_<COIN>/` | `meta.json`(커밋·데이터 sha256·설정) `probe.csv` `daily.csv` `summary.csv` |

## 시뮬레이터 가정 — 결과는 pess/prop 두 값의 범위로 읽고, 판정은 pess로 한다
- 우리 주문은 유령이다(장부·체결을 바꾸지 않음). 주문·취소 지연 `lat_ms`=200, 취소는 같은 ms의 체결보다 먼저 처리.
- 수동 체결: 우리 가격보다 나쁜 가격의 체결 → 그 체결량만큼 체결(관통). 같은 가격 → 앞줄(ahead)이 소진된 뒤 체결. 장부가 우리 가격을 넘어오면 전량 체결.
- 앞줄: pess = 합류 시 보이는 잔량 전부가 앞, 취소는 전부 뒤에서. prop = 취소를 비례 배분.
- 보이는 20레벨 밖: pess = 관통 체결만, prop = 가장 깊은 보이는 레벨의 잔량을 앞줄로.
- 한쪽으로 흐름이 기준선의 `freeze`배(기본 1)를 넘는 동안에는 그쪽 주문을 시장에서 멀어지는 방향으로 옮기지 않는다(깊은 주문이 기다리던 스윕을 피해 달아나지 않게).
- 재고를 줄이는 쪽 호가는 정규 호가와 '평균 진입가 + `d_exit`'(익절) 중 시장에 가까운 것. 움직이는 중간가가 아니라 진입가에 묶는다. `t_hold` 초과 시 테이커로 정리.
- ALO: 활성화 시점에 반대편과 교차하면 거부. 테이커 털기: 보이는 호가를 걸어 VWAP + `slip_bp`, 보이는 깊이 밖은 +10bp.
- 수수료 메이커 1.5 / 테이커 4.5 bp(HL 기본 등급). 펀딩 무시. 하루 끝에 재고 0. 처음 `warm_s`=900초는 특징 예열(주문 없음).
- 상태 x(쪽별): 청산 흐름비·비청산 공격 흐름비(2초 EWMA / 10분 기준선), 1분 변동성, 1호가 불균형, 1분 중간가 이탈.
- 추정기 (a) 기준가 = 중간가 + β·(불균형1, 불균형5)·반스프레드, 5초 뒤 중간가에 회귀. (b) 체결 조건부 60초 순마크아웃(수수료 후)을 깊이·x로 릿지. 학습은 직전 `--train`일의 탐침 체결만 쓴다. 특징은 체결 시각보다 `lat_ms` 이상 앞선 마지막 결정 시점의 값.

## 규칙
- 데이터와 결과는 git 밖(`.gitignore`). 결과마다 `meta.json`에 커밋 해시와 데이터 해시가 남는다. `dirty`인 결과는 판정에 쓰지 않는다.
- 시도한 설정 수가 DSR의 N이다. 실행 간에 누적되므로 프리셋·그리드를 바꿀 때마다 기록한다.
- 노드 서버에서는 저장소를 `~/hl` 밖에 둔다(`hl-node` 기본 데이터 경로가 `~/hl/data`).
