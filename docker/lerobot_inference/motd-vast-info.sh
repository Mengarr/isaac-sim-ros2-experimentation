#!/bin/bash
# Dynamic MOTD: show the public IPv4 and the external zenoh (7447) port so you
# can connect to the instance at a glance on SSH login.
# Runs via /etc/update-motd.d/ (pam_motd), after pam_env has loaded
# /etc/environment, so VAST_* vars are available here.

ipv4=$(curl -4 -s --max-time 3 ifconfig.me 2>/dev/null)
zenoh_port=${VAST_TCP_PORT_7447:-<not mapped>}

echo
echo "=== Instance connection info ==="
echo "  Public IPv4 : ${ipv4:-<unavailable>}"
echo "  Zenoh 7447  : ${ipv4:-<ip>}:${zenoh_port}  (container :7447)"
echo "================================"
echo
