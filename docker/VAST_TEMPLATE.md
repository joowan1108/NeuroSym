# Docker 이미지와 Vast.ai 템플릿

## 1. 이미지 빌드와 푸시 (내 PC)

Docker Hub 에 `joowan1108/dsrl-robocasa365` 저장소를 **Private 으로 먼저 만든 뒤** 푸시합니다
(저장소 없이 푸시하면 public 으로 자동 생성됩니다).

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t joowan1108/dsrl-robocasa365:v1 .
docker login -u joowan1108
docker push joowan1108/dsrl-robocasa365:v1
```

이미지는 패키지 구성(`setup/install_*.sh`, `setup/env.sh` 의 버전 고정)이 바뀔 때만 다시 만듭니다.
코드 수정은 GitHub 에 push 하고 인스턴스에서 `git pull` 합니다.

## 2. 템플릿 (cloud.vast.ai/templates → 새 템플릿)

| 항목 | 값 |
|---|---|
| Template Name | `joowan-dsrl-robocasa365-v1` |
| Image Path:Tag | `joowan1108/dsrl-robocasa365:v1` |
| Docker Options / Ports | `-e NVIDIA_DRIVER_CAPABILITIES=all` (포트 추가 없음) |
| Environment Variables | `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONUNBUFFERED=1` |
| Launch Mode | SSH |
| On-start Script | 아래 스크립트 (`setup/onstart.sh` 와 동일) |
| Docker Repository Authentication | 사용자명 `joowan1108` + Docker Hub **Read-only** 액세스 토큰, 서버 이름은 비움 |
| Disk Space | 150 GB |
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
- 토큰은 템플릿이 아니라 cloud.vast.ai/account → Environment Variables 에 넣습니다.
  `HF_TOKEN` (RLDX-1 체크포인트가 인증을 요구할 때), `GITHUB_TOKEN` (저장소를 private 으로 바꿀 때만).

## 3. 인스턴스 대여

필터: RTX 4090 ×1, On-demand, **Max CUDA ≥ 12.4**, vCPU 16 이상, RAM 64 GB 이상,
다운로드 500 Mbps 이상, Reliability 99% 이상.

접속 후 순서는 README 의 "vast.ai 에서 실행" 을 따릅니다. 끝나면 `/workspace/results` 를 내려받고 Destroy 합니다.

## 4. Vast 에 올리기 전 확인 (GPU 가 있는 서버에서)

```bash
docker run --rm -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd)/setup/onstart.sh:/test/onstart.sh \
  --entrypoint bash joowan1108/dsrl-robocasa365:v1 \
  -c "bash /test/onstart.sh && tail -f /workspace/download.log"
```
