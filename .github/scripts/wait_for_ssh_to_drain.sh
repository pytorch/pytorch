#!/usr/bin/env bash

set -eou pipefail

echo "Holding runner for 2 hours until all ssh sessions have logged out"
for _ in $(seq 1440); do
    # Break if no ssh session exists anymore
    if [ "$(who)" = "" ]; then
      break
    fi
    echo "."
    sleep 5
done
