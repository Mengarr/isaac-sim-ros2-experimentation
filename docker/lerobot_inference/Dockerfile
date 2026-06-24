# Image: mengar/lerobot_inference
# Base: Ubuntu 24.04 server, ROS 2 Jazzy (base) + zenoh rmw, uv + HF CLI baked in.
#
# Everything that is slow/stable (apt, ROS, uv, hf) is baked into the image.
# Repos + the lerobot uv venv are set up at RUNTIME into the persistent
# /workspace volume by entrypoint.sh (guarded so it only runs once).
#
# Build:  docker build -t mengar/lerobot_inference -f Dockerfile .
# Push:   docker push mengar/lerobot_inference
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
# ROS 2 Jazzy (base) + dev tools + zenoh rmw
# ---------------------------------------------------------------------------
RUN set -euxo pipefail \
    && export ROS_APT_SOURCE_VERSION="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F 'tag_name' | awk -F'"' '{print $4}')" \
    && source /etc/os-release \
    && UBUNTU_CODE="${UBUNTU_CODENAME:-${VERSION_CODENAME}}" \
    && curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODE}_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-dev-tools \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-demo-nodes-py \
        ros-jazzy-rmw-zenoh-cpp \
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
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc \
    && echo 'export RMW_IMPLEMENTATION=rmw_zenoh_cpp' >> /root/.bashrc \
    && echo 'export HF_HOME=/workspace/.cache' >> /root/.bashrc \
    && echo '[ -d /workspace ] && cd /workspace' >> /root/.bashrc \
    && echo '[ -f /workspace/repos/lerobot/.venv/bin/activate ] && source /workspace/repos/lerobot/.venv/bin/activate' >> /root/.bashrc

# ---------------------------------------------------------------------------
# Runtime init: clones repos + sets up the uv venv into the persistent
# /workspace volume on first launch (no-op on subsequent launches).
# ---------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Dynamic MOTD: print public IPv4 + external zenoh port on SSH login.
COPY motd-vast-info.sh /etc/update-motd.d/99-vast-info
RUN chmod +x /etc/update-motd.d/99-vast-info

# Documented ports (actual publishing is done via Vast.ai docker options / -p):
#   22   ssh
#   7447 zenoh router tcp endpoint
EXPOSE 22 7447

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
