# Image: mengar/lerobot_train
# Base: Ubuntu 24.04 server, uv + HF CLI baked in.
#
# Everything that is slow/stable (apt, uv, hf) is baked into the image.
# Repos + the lerobot uv venv are set up at RUNTIME into the persistent
# /workspace volume by entrypoint.sh (guarded so it only runs once).
#
# Build:  docker build -t mengar/lerobot_train -f Dockerfile .
# Push:   docker push mengar/lerobot_train
FROM ubuntu:24.04

# Run as root during build; no `sudo` needed inside the image.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Base system tooling
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        gnupg \
        git \
        build-essential \
        python3-venv \
        unzip \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv (Python env manager) + Hugging Face CLI  (installed to /root/.local/bin,
# i.e. baked into the image, outside the /workspace volume)
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# AWS CLI v2
# ---------------------------------------------------------------------------
RUN curl -L "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# Environment for interactive (SSH) shells. Set here rather than via docker
# `-e`, because Vast SSH sessions spawn a fresh login shell that does NOT
# inherit the container PID 1 environment.
RUN echo 'export HF_HOME=/workspace/.cache' >> /root/.bashrc

# ---------------------------------------------------------------------------
# Runtime init: clones repos + sets up the uv venv into the persistent
# /workspace volume on first launch (no-op on subsequent launches).
# ---------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Documented ports (actual publishing is done via Vast.ai docker options / -p):
#   22   ssh
EXPOSE 22

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
