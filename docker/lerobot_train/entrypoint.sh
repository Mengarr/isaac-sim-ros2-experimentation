#!/usr/bin/env bash
#
# Runtime init for mengar/lerobot_train.
#
# /workspace is a persistent volume (Vast.ai mounts it per-instance). On the
# FIRST launch it is empty, so we clone the repos and build the lerobot uv
# venv into it. On every subsequent stop/start the volume already has
# everything, so the guards make this a near-instant no-op.
set -euo pipefail

WS=/workspace
REPOS="$WS/repos"
LEROBOT="$REPOS/lerobot"
EXP="$REPOS/isaac-sim-ros2-experimentation"

log() { echo "[entrypoint] $*"; }

# --- expose container env to SSH shells ----------------------------------
# Vast injects vars (VAST_TCP_PORT_*, etc.) into PID 1's environment, but SSH
# login shells do NOT inherit it. Persisting them to /etc/environment makes
# PAM (pam_env) load them for every SSH session -- this is what Vast's own
# base images do for you. We run on-start with the full container env, so
# capture it here.
log "Persisting container env to /etc/environment for SSH shells..."
while IFS= read -r -d '' line; do
    case "$line" in
        # skip noise / vars that should not be globally pinned
        PWD=*|OLDPWD=*|SHLVL=*|HOME=*|_=*|HOSTNAME=*|TERM=*) continue ;;
    esac
    name=${line%%=*}
    # de-dupe: drop any prior definition, then append the current one
    sed -i "/^${name}=/d" /etc/environment 2>/dev/null || true
    printf '%s\n' "$line" >> /etc/environment
done < /proc/1/environ

# Persistent directory layout
mkdir -p "$REPOS" "$WS/data/models" "$WS/data/datasets" "$WS/.cache"

# --- repos ---------------------------------------------------------------
if [ ! -d "$LEROBOT/.git" ]; then
    log "Cloning lerobot..."
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT"
else
    log "lerobot already present, skipping clone."
fi

if [ ! -d "$EXP/.git" ]; then
    log "Cloning isaac-sim-ros2-experimentation..."
    git clone https://github.com/Mengarr/isaac-sim-ros2-experimentation.git "$EXP"
else
    log "isaac-sim-ros2-experimentation already present, skipping clone."
fi

# --- lerobot uv venv -----------------------------------------------------
# `.venv` lives inside the lerobot repo (under /workspace), so it persists too.
if [ ! -d "$LEROBOT/.venv" ]; then
    log "Setting up lerobot uv venv (uv sync + pip install -e .[pi,dataset,training] + peft)..."
    pushd "$LEROBOT" >/dev/null
    uv sync
    uv pip install -e ".[pi,dataset,training,sarm]"
    uv pip install peft
    popd >/dev/null
else
    log "lerobot uv venv already present, skipping setup."
fi

log "Workspace ready."

# When called as `entrypoint.sh --init-only` (e.g. from a Vast.ai on-start
# script in SSH launch mode) just do the init and return. Otherwise hand off
# to the container command (default: bash) with `exec` so signals / PID 1
# behave correctly (Docker ENTRYPOINT launch mode).
if [ "${1:-}" = "--init-only" ]; then
    exit 0
fi
exec "$@"
