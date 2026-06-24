#!/bin/bash
# Dynamic MOTD: show the public IPv4 and the external zenoh (7447) port so you
# can connect to the instance at a glance on SSH login.
# Runs via /etc/update-motd.d/ (pam_motd) through run-parts, whose child
# processes inherit sshd's environment -- NOT the pam_env session environment.
# So /etc/environment vars (incl. VAST_TCP_PORT_*) are NOT auto-available here;
# we must source the file ourselves.
if [ -r /etc/environment ]; then
    set -a
    . /etc/environment
    set +a
fi

ipv4=$(curl -4 -s --max-time 3 ifconfig.me 2>/dev/null)
zenoh_port=${VAST_TCP_PORT_7447:-<not mapped>}

echo
echo "=== Instance connection info ==="
echo "  Public IPv4 : ${ipv4:-<unavailable>}"
echo "  Zenoh 7447  : ${ipv4:-<ip>}:${zenoh_port}  (container :7447)"
echo "================================"
echo
