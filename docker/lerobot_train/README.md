# lerobot_train — Vast.ai image

Image: `mengar/lerobot_train` (Ubuntu 24.04, `uv` + HF CLI baked in).

Training counterpart to `lerobot_inference`: no ROS 2 / zenoh, and the
lerobot venv is installed with the training extras plus `peft`.

Slow/stable software is baked into the image. Repos and the lerobot `uv` venv are
set up at runtime into `/workspace` (a volume you mount via Vast) by
`entrypoint.sh`, guarded so first launch initializes and relaunches are no-ops.

## Build & push

Run from this directory (the build context is `.` so the Dockerfile can
`COPY entrypoint.sh`).

```bash
# 1. Log in to Docker Hub (use a Hub access token as the password)
docker login -u mengar

# 2. Build (also tag a version alongside :latest if you want to roll back)
docker build -t mengar/lerobot_train:latest -f Dockerfile .

# 3. Push
docker push mengar/lerobot_train:latest
docker push mengar/lerobot_train:v1
```

## Vast.ai template config

**Launch mode:** SSH (direct or proxy).

**On-start script:**

```bash
#!/bin/bash
/usr/local/bin/entrypoint.sh --init-only
```

Clones `lerobot` + `isaac-sim-ros2-experimentation` into `/workspace/repos`,
runs `uv sync` + `uv pip install -e ".[pi,dataset,training]"` + `uv pip install
peft`, and creates `/workspace/data/{models,datasets}`. Skips anything already
present.

**Docker options:**

```
-p 22:22
```

- `22` — ssh

`HF_HOME=/workspace/.cache` is exported from `/root/.bashrc` in the image,
**not** via docker `-e`. Vast SSH sessions spawn a fresh login shell that does
not inherit the container PID 1 environment, so `-e` vars are invisible in your
shell; the `.bashrc` exports always apply.

**Container disk size:** ~20 GB.

> Only needs to hold the image (~3–4 GB) plus scratch space. The venv, model
> weights, and datasets live on the `/workspace` volume (sized separately).

## GPU / CUDA

The image is based on plain `ubuntu:24.04` and bakes in **no CUDA toolkit**. GPU
support comes entirely from the **PyTorch wheel** that `uv`/lerobot installs —
modern torch wheels bundle their own CUDA runtime libraries (cuDNN, cuBLAS,
etc.). The only thing required from the host is the **NVIDIA driver**, which Vast
injects into the container via the NVIDIA Container Toolkit.

The host driver version sets a ceiling on the CUDA version a container can use
(drivers are backward compatible with older CUDA, not newer). So when selecting a
Vast host, **pick one whose NVIDIA driver is recent enough for the CUDA version
torch targets** (currently CUDA 12.x). If `torch.cuda.is_available()` returns
`False` on a host where install succeeded, the driver is too old — choose a
different host rather than changing the image.

## Logs
Check if the entrypoint script was successful with `cat /var/log/onstart.log`
