#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
mkdir -p deployment/logs .cache/downloads .cache/compat
trap 'echo "failed" > deployment/status.txt' ERR
echo building > deployment/status.txt
# Credentials remain in the pre-existing private server configuration.
source /home/ljr/.config/mihomo-private/proxy.env
export DOCKER_BUILDKIT=1
export NO_PROXY="${NO_PROXY:-},.nvidia.com,.nvidia.cn"
export no_proxy="$NO_PROXY"
if [ ! -d IsaacLab/.git ]; then
 git clone --filter=blob:none --no-checkout https://github.com/isaac-sim/IsaacLab.git IsaacLab
 git -C IsaacLab checkout 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
fi
docker build --network host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY --build-arg NO_PROXY --build-arg no_proxy -f IsaacLab/docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 IsaacLab
docker build --network host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY --build-arg NO_PROXY --build-arg no_proxy -t supermimic:latest docker/
echo compatibility > deployment/status.txt
curl -fL --retry 4 -C - -o .cache/downloads/cuda-compat-13-0.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-0_580.65.06-0ubuntu1_amd64.deb
dpkg-deb -x .cache/downloads/cuda-compat-13-0.deb .cache/compat
# Forward compatibility is confined to this project's container; host drivers stay intact.
docker run -d --name supermimic --gpus '"device=0"' --memory=128g --shm-size=16g --user "$(id -u):$(id -g)" \
 -e HOME=/workspace/supermimic/.cache/home \
 -e LD_LIBRARY_PATH=/workspace/supermimic/.cache/compat/usr/local/cuda-13.0/compat \
 -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp \
 -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
 -e XDG_CACHE_HOME=/workspace/supermimic/.cache \
 -e MPLCONFIGDIR=/workspace/supermimic/.cache/matplotlib \
 -v "$ROOT":/workspace/supermimic -w /workspace/supermimic supermimic:latest sleep infinity
docker exec -u root supermimic groupadd -g "$(id -g)" ljr
docker exec -u root supermimic useradd -M -u "$(id -u)" -g "$(id -g)" -d /workspace/supermimic/.cache/home ljr
echo downloading_data > deployment/status.txt
bash tools/download_data.sh
echo checking > deployment/status.txt
docker exec supermimic python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); x=torch.ones(8,device="cuda"); print(x.sum().item())'
docker exec supermimic python deployment/verify_runtime.py > deployment/logs/compile.log 2>&1
docker exec supermimic python tools/test_env.py > deployment/logs/test_env.log 2>&1
docker exec supermimic python deployment/verify_render.py > deployment/logs/render.log 2>&1
docker exec supermimic python tools/render.py results/verification_reference.npz results/verification_reference.mp4 --seconds 3 --title "REFERENCE ONLY - environment check" >> deployment/logs/render.log 2>&1
echo ready > deployment/status.txt
