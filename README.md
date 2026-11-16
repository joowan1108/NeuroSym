# DSRL on RoboCasa365 — RLDX-1 / Xiaomi-Robotics-1

논문 *Steering Your Diffusion Policy with Latent Space Reinforcement Learning* (DSRL, arXiv 2506.15799)을
두 개의 RoboCasa365 파인튜닝 VLA에 적용해 3개 태스크에서 성능을 측정하는 코드입니다.

| 모델 | 체크포인트 | 액션 헤드 | 조종하는 잠재 노이즈 |
|---|---|---|---|
| `xr1` | `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` | flow matching, Euler 5 step | `(16, 60)` 중 유효 12차원 |
| `rldx` | `RLWRLD/RLDX-1-FT-RC365` | flow matching (MSAT) | `(16, 64)` 전체 64차원 |

태스크: `CoffeeSetupMug`(horizon 600), `GetToastedBread`(3000), `SteamInMicrowave`(2100).

## 방법 요약

DSRL은 VLA 가중치를 건드리지 않고, flow matching의 **초기 노이즈 w** 를 고르는 작은 정책 π^W(s)를
RL로 학습합니다. 여기서는 논문의 π0 실험 설정(부록 C.5, 표 11)인 **DSRL-SAC** 를 그대로 따릅니다.

- 잠재 액션: 청크 전체의 초기 노이즈 `w ∈ [-b_W, b_W]^(16×12)` 를 직접 조종 (`b_W = 3.0`, `--noise-mode chunk`).
  논문의 π0 방식(스텝 노이즈 하나를 청크 축으로 반복, `--noise-mode repeat`)은 XR-1 에서 원본 성공률을
  50% → 0/20 으로 떨어뜨려 기본값에서 뺐습니다. 액터는 w ~ N(0, I), 즉 원본 정책과 같은 분포에서 출발합니다.
  실행 전에 w 의 크기를 √d 로 맞춥니다(방향만 조종). 크기가 작은 노이즈는 XR-1 성공률을 15~20% 로 떨어뜨리는데,
  SAC 가 학습하면서 노이즈를 평균 쪽으로 줄여 실제로 그 문제가 생겼기 때문입니다(학습 성공률 41% → 6%).
- 관측: 카메라 3대 64×64 이미지(4층 CNN) + 고유감각 16차원
- 보상: 성공 전까지 매 결정마다 −1, 성공 시 0 후 종료 / γ = 0.999
- SAC: 크리틱 10개 평균, target entropy −d/2, UTD 20, batch 256, actor lr 1e-4, critic lr 3e-4
- SAC 초기 온도는 0.01 (`--init-alpha`). 1.0으로 두면 엔트로피 보너스가 "에피소드를 끝내지 않는 것"을
  보상해서 성공률이 떨어지는 것을 토이 환경에서 확인했습니다. 논문에 없는, 이 구현의 선택입니다.
- 워밍업: w ~ N(0, I) 로 base 정책 8 에피소드를 굴려 버퍼 초기화
- 액션 청크 하나를 하나의 액션으로 취급 (xr1: 16 스텝 실행, rldx: 8 스텝 실행 — 각 공식 평가 기본값)

노이즈 주입은 두 모델의 샘플링 코드가 호출하는 `torch.randn_like` / `torch.randn` 을 호출 순간에만
바꿔치기하는 방식입니다(`dsrl/noise_hook.py`). 주입이 실제로 일어나지 않으면 서버가 에러를 냅니다.

## 측정 항목

`scripts/summarize.py` 가 `$RESULTS/summary.csv` 와 표를 만듭니다.

| 항목 | 정의 |
|---|---|
| base SR | 원본 정책(매 청크 w ~ N(0, I)) 성공률, 50 에피소드 |
| DSRL SR | 학습된 π^W 의 결정적 정책 성공률, 같은 50개 시드(같은 장면) |
| RL time (s) | RL에 쓴 벽시계 시간 = 롤아웃 + 그래디언트 업데이트 (워밍업 포함, 중간 평가 시간 제외) |
| RL env steps (#) | RL이 소비한 환경 스텝 수 (워밍업 포함) |
| RL VLA queries (#) | VLA 호출 횟수 = 잠재 MDP 전이 수 |
| time/steps to best | 중간 평가에서 최고 성공률에 처음 도달한 시점까지의 RL 시간·스텝 |
| steps/episode | 성공 에피소드의 평균 길이 (논문 그림 17의 steps-to-success) |
| wall time all stages (h) | 태스크 하나의 base 평가 + 학습(중간 평가 포함) + DSRL 평가 벽시계 시간 |

GPU 사용량은 모델별로 `$RESULTS/gpu_summary.csv` 에 따로 정리됩니다.

| 항목 | 정의 |
|---|---|
| GPU | `nvidia-smi` 가 보고한 GPU 이름 (`$RESULTS/gpu.txt`) |
| GPU hours (h) | 모델 서버를 띄운 순간부터 내린 순간까지의 시간 (다운로드 대기 제외) |
| model loading (h) / GPU hours excl. loading (h) | 그중 서버가 `ready` 를 찍기까지의 모델 로딩 시간과, 그것을 뺀 실제 실험 시간 |
| mean GPU util (%) / peak GPU memory (MiB) | 10초마다 기록한 `$RESULTS/<model>/gpu_usage.csv` 의 평균 사용률과 최대 메모리 |

마지막 줄에 두 모델 합계가 로딩 포함/제외로 나옵니다.

`train_summary.json` 에는 시간 분해(VLA 추론 / 시뮬레이터 / SAC 업데이트)와 중간 평가 곡선도 들어 있습니다.
`--target-success 0.8` 을 주면 그 성공률에 처음 도달한 시간·스텝(`to_target`)도 기록합니다.

## vast.ai 에서 실행

환경은 Docker 이미지에, 코드는 이 저장소에, 에셋·체크포인트는 인스턴스에서 받습니다.
템플릿 설정값과 이미지 빌드/푸시 방법은 [docker/VAST_TEMPLATE.md](docker/VAST_TEMPLATE.md) 에 있습니다.

| 위치 | 내용 |
|---|---|
| 이미지 `/opt/venvs/sim`, `/opt/venvs/xr1`, `/opt/src/RLDX-1/.venv` | 시뮬레이터+학습 / XR-1 / RLDX-1 파이썬 환경 |
| 이미지 `/opt/src` | 고정된 버전의 robosuite, robocasa, Xiaomi-Robotics-1, RLDX-1 |
| 인스턴스 `/workspace/NeuroSym` | 이 저장소 (On-start Script 가 clone) |
| 인스턴스 `/workspace/checkpoints`, `/workspace/results` | 체크포인트(24 GB), 결과 |

GPU 는 **RTX 4090** 기준입니다(이미지의 RLDX-1 환경이 공식 lockfile 의 CUDA 12.6 빌드 torch 를 씁니다).

On-start 가 에셋 → XR-1 체크포인트 → RLDX-1 체크포인트 순서로 백그라운드 다운로드를 시작합니다.
아래 명령은 다운로드가 끝나기를 기다리지 않고 접속하자마자 실행해도 됩니다. 각 명령이 자기에게
필요한 것만 기다리므로(`check.sh` 와 XR-1 은 에셋과 XR-1 체크포인트, RLDX-1 은 RLDX-1 체크포인트),
XR-1 실험이 도는 동안 RLDX-1 체크포인트가 받아집니다. 다운로드가 실패하면 명령이 바로 멈추고 알려 줍니다.

```bash
# 인스턴스 접속 후 (진행 상황: tail -f /workspace/download.log)
cd /workspace/NeuroSym && tmux new -s dsrl
bash setup/check.sh                      # GPU, EGL 렌더링, 시뮬레이터 속도 확인

# 먼저 짧게 확인 (모델 하나, 태스크 하나)
MODELS=xr1 TASKS=CoffeeSetupMug EVAL_EPISODES=4 STEPS_CoffeeSetupMug=5000 \
  RESULTS=/workspace/results_smoke bash scripts/run_all.sh

# 전체 실험 (결과: /workspace/results)
bash scripts/run_all.sh
```

`run_all.sh` 는 모델별로 서버를 띄우고 태스크마다 base 평가 → DSRL 학습 → DSRL 평가를 실행합니다.
끝난 단계는 건너뛰므로 중단되면 다시 실행하면 됩니다.

개별 실행:

```bash
source setup/env.sh && export PYTHONPATH=$PROJECT
bash scripts/run_server.sh rldx 10086 &          # 모델 서버 (모델 전용 환경)
PY=$VENVS/sim/bin/python                         # 시뮬레이터 + 학습 환경
OUT=$RESULTS/rldx/SteamInMicrowave
$PY -m dsrl.eval --model rldx --task SteamInMicrowave --out-dir $OUT
$PY -m dsrl.train_dsrl --model rldx --task SteamInMicrowave --total-env-steps 200000 \
    --out-dir $OUT
$PY -m dsrl.eval --model rldx --task SteamInMicrowave --agent $OUT/best.pt --out-dir $OUT
```

주요 옵션: `--n-envs`(병렬 환경 수), `--split {pretrain,target}`, `--horizon`, `--exec-steps`,
`--magnitude`(b_W), `--utd`, `--eval-every`, `--eval-episodes`, `--target-success`.
`dsrl.eval --noise repeat` 은 "청크 축 반복 노이즈" 자체가 base 성능을 해치지 않는지 확인하는 용도입니다.

## 구조

```
dsrl/servers/xr1_server.py   XR-1 서버: 공식 entry.py 의 전/후처리를 재사용 + 노이즈 주입
dsrl/servers/rldx_server.py  RLDX-1 서버: 공식 RLDXSimPolicyWrapper 파이프라인 + 노이즈 주입
dsrl/noise_hook.py           초기 노이즈 주입 훅
dsrl/wire.py                 서버-클라이언트 RPC (환경 간 numpy 버전이 달라 pickle 미사용)
dsrl/envs.py                 RoboCasa365 워커 프로세스 풀 (청크 단위 스텝)
dsrl/rollout.py              잠재 MDP 롤아웃 루프
dsrl/sac.py                  DSRL-SAC 에이전트, 리플레이 버퍼
dsrl/train_dsrl.py           학습 + 시간/스텝 측정
dsrl/eval.py                 base / DSRL 평가
```

## 알아둘 점

- 기본 평가 조건은 두 모델 동일하게 `split=pretrain`, RoboCasa 공식 horizon, 시드 고정입니다.
  Xiaomi 공식 평가와 같은 조건이고, RLDX 공식 스크립트 기본값(`split=target`, horizon 400/2000/1400)과는 다릅니다.
- RL 예산 기본값은 100k / 300k / 200k 환경 스텝입니다(`STEPS_<Task>` 로 변경).
- RLDX-1 체크포인트는 비상업 라이선스(RLWRLD Model License v1.0)입니다.
