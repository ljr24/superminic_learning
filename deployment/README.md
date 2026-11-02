# 196 server deployment

Project: `/home/ljr/LJR_projects/super_minic` on `lab196` (`ljr@cdmc-a100`).

The code tracks `origin/main` at https://github.com/ljr24/superminic_learning.git. Isaac Lab is pinned to `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`.

## Installation

Run `bash deployment/setup_server.sh` from the repository. The script expects the existing private proxy configuration on this server. It does not contain credentials. A fresh installation requires the container name `supermimic` to be available.

The pinned dependencies use CUDA 13. The server has driver 570.172.08. NVIDIA's CUDA 13 forward compatibility package is extracted under `.cache/compat` and selected only within this container; the host driver is unchanged. This configuration must pass the GPU and simulation checks before use.

## Locations

- `IsaacLab/`: pinned external source checkout (ignored by Git)
- `data/`: motions, G1 model and converted assets (ignored)
- `.cache/`: compatibility libraries, compilers and runtime caches (ignored)
- `logs/`: training runs/checkpoints (ignored)
- `results/`: visual results; MP4 and NPZ are ignored
- `deployment/logs/`: installation and verification logs (ignored)
- `deployment/status.txt`: installation stage; `ready` means the setup script completed its checks
- Docker images and build layers: server-managed `/var/lib/docker`

## Connect and inspect

```bash
ssh lab196
cd ~/LJR_projects/super_minic
cat deployment/status.txt
tail -n 30 deployment/logs/setup.log
docker exec -it supermimic bash
```

Training is not started by installation. Check current GPU utilization and coordinate GPU allocation before training. The initial container exposes physical GPU 0 only; adjust its GPU selection before a real run if necessary.

```bash
# After selecting an available GPU and completing setup:
./run_train.sh lafan1_main
# Training lives in tmux inside the container.
docker exec -it supermimic tmux attach -t lafan1_main
```

## Sync

Commit and push local code first, then on the server:

```bash
cd ~/LJR_projects/super_minic
git pull --ff-only origin main
```

Do not update code during a running experiment. Models, datasets and experiment outputs are managed separately from Git. Never commit proxy credentials or private keys.
