#!/bin/bash
set -eu
: "${IDLE_TIMEOUT:?IDLE_TIMEOUT must be set}"

export -p > /tmp/.lumen_env
echo "cd $PWD && source /tmp/.lumen_env" >> /root/.bashrc
echo 'trap "touch /tmp/.lumen_done" EXIT' >> /root/.bashrc

echo ""
echo "=== Job finished. Container idle for $((IDLE_TIMEOUT / 60))m. ==="
echo "Attach: kubectl exec -it $HOSTNAME -n remote-execution-job-space-beta -- bash"
echo "Extend: echo 0 > /tmp/.lumen_idle_seconds"
echo ""

MAX_TIMEOUT=14400  # 4 hours hard cap
SECONDS=0
echo 0 > /tmp/.lumen_idle_seconds
while [ ! -f /tmp/.lumen_done ]; do
    sleep 10
    if [ "$SECONDS" -ge "$MAX_TIMEOUT" ]; then
        echo "Hard timeout (4h) reached."
        break
    fi
    CURRENT=$(cat /tmp/.lumen_idle_seconds)
    echo $((CURRENT + 10)) > /tmp/.lumen_idle_seconds
    if [ "$((CURRENT + 10))" -ge "$IDLE_TIMEOUT" ]; then
        echo "Idle timeout reached."
        break
    fi
done
echo "Session ended."
