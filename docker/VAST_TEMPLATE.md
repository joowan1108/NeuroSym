# Docker 이미지와 Vast.ai 템플릿

연구실 Vast.ai 가이드의 순서(이미지 → Docker Hub → 동작 확인 → 템플릿 → 대여 → 종료)를 따릅니다.

| 무엇 | 어디에 |
|---|---|
| OS 패키지, CUDA 12.4 devel, 파이썬 환경 3개, 고정된 외부 저장소 | Docker 이미지 `/opt` |
| 이 저장소의 코드 | GitHub `joowan1108/NeuroSym` (Public, On-start 에서 clone) |
| RoboCasa 에셋 (~10 GB), 체크포인트 (~24 GB) | On-start 에서 인스턴스로 다운로드 |
| `HF_TOKEN` 등 토큰 | Vast 계정 환경변수 |

## 1. 이미지 빌드와 Docker Hub 푸시 (내 PC)

Docker Hub 에 `joowan1108/dsrl-robocasa365` 저장소를 **Private 으로 먼저 만든 뒤** 푸시합니다
(저장소 없이 푸시하면 public 으로 자동 생성됩니다). 푸시 후 저장소 페이지에 Private 표시가 있는지 확인합니다.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t joowan1108/dsrl-robocasa365:v1 .
docker login -u joowan1108
docker push joowan1108/dsrl-robocasa365:v1
```

- 이미지는 패키지 구성(`setup/install_*.sh`, `setup/env.sh` 의 버전 고정)이 바뀔 때만 다시 만듭니다.
  코드 수정은 GitHub 에 push 하고 인스턴스에서 `git pull` 합니다.
- 이미지가 큽니다(베이스 위에 환경 3개가 약 26 GB). 인스턴스가 처음 뜰 때 Loading 이 길 수 있습니다.

pull 용 액세스 토큰: Docker Hub → Account Settings → Personal access tokens → Generate,
권한은 **Read-only** (`Public Repo Read-only` 는 private 이미지를 받지 못함). 생성 직후 한 번만 보이니 바로 복사합니다.

## 2. 동작 확인 (Vast 에 올리기 전)

On-start Script 를 깨끗한 컨테이너에서 그대로 실행해 봅니다.

```bash
docker run --rm -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd)/setup/onstart.sh:/test/onstart.sh \
  --entrypoint bash joowan1108/dsrl-robocasa365:v1 \
  -c "bash /test/onstart.sh && tail -f /workspace/download.log"
```

GPU 없는 PC 에서 `--gpus all` 을 빼고 확인한 항목: GitHub clone, 에셋 다운로드가 프롬프트 없이 진행됨,
경로 설정, 세 환경에서 이 저장소 코드 import. GPU 렌더링과 모델 로딩은 Vast 에서 `setup/check.sh` 로 처음 확인합니다.

## 3. 템플릿 (cloud.vast.ai/templates → 새 템플릿, 팀 계정에서)

| 항목 | 값 |
|---|---|
| Template Name | `joowan-dsrl-robocasa365-v1` |
| Image Path:Tag | `joowan1108/dsrl-robocasa365:v1` |
| Docker Options / Ports | `-e NVIDIA_DRIVER_CAPABILITIES=all` (포트 추가 없음) |
| Environment Variables | `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONUNBUFFERED=1` |
| Launch Mode | SSH |
| On-start Script | 아래 스크립트 (`setup/onstart.sh` 와 동일) |
| Docker Repository Authentication | 사용자명 `joowan1108` + 1장의 Read-only 토큰, 서버 이름은 비움 |
| Disk Space | 150 GB (이미지 + 에셋 + 체크포인트 + 결과) |
| Template Visibility | Private |

On-start Script:

```bash
cd /workspace
if [ ! -d NeuroSym ]; then
  git clone "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/joowan1108/NeuroSym.git"
fi
cd NeuroSym && git pull --ff-only || true
nohup bash setup/download.sh > /workspace/download.log 2>&1 &
```

- `NVIDIA_DRIVER_CAPABILITIES=all` 은 MuJoCo 의 GPU(EGL) 렌더링에 필요합니다. 이미지에도 들어 있지만
  컨테이너 생성 시점에 적용되도록 템플릿에도 넣습니다.
- 토큰은 템플릿이 아니라 Account → Environment Variables 에 넣습니다. `HF_TOKEN` 은 RLDX-1 체크포인트가
  인증을 요구할 때 필요합니다. 저장소가 Public 이라 `GITHUB_TOKEN` 은 필요 없습니다.
- **SSH 키는 개인 계정(Personal view)에서만 등록할 수 있습니다.** 개인 계정의 SSH Keys 에
  `~/.ssh/id_ed25519.pub` 한 줄을 등록한 뒤 팀 계정으로 돌아와 대여합니다.

## 4. 인스턴스 대여와 사용 (팀 계정에서)

필터: RTX 4090 ×1, On-demand, **Max CUDA ≥ 12.4**, 디스크 150 GB, vCPU 16 이상, RAM 64 GB 이상,
다운로드 500 Mbps 이상, Reliability 99% 이상.

Rent → Loading(이미지 다운로드) → Running 이 되면 SSH 로 접속합니다.
접속이 거부되면 인스턴스 카드의 열쇠 아이콘에서 그 인스턴스에 SSH 키를 추가합니다.

```bash
cd /workspace/NeuroSym && bash setup/check.sh   # nvidia-smi, 세 환경의 CUDA, EGL 렌더링, 시뮬레이터 속도
```

다운로드가 끝나기를 기다릴 필요 없습니다. `check.sh` 와 `run_all.sh` 는 필요한 파일이 받아질 때까지
스스로 기다리고, XR-1 실험이 도는 동안 RLDX-1 체크포인트를 받습니다(진행 상황: `tail -f /workspace/download.log`).
이후 실험 순서는 README 의 "vast.ai 에서 실행" 을 따릅니다.

## 5. 종료

1. `/workspace/results` 를 내려받거나 Hugging Face Hub 등에 업로드합니다.
2. 인스턴스를 **Destroy** 합니다. Stop 은 GPU 요금만 멈추고 디스크 요금은 계속 나갑니다.
   다시 쓸 때는 템플릿으로 새로 띄웁니다.
