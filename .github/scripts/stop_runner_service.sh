#!/bin/bash

set +e
set -x

# Get the service name
RUNNER_SERVICE=$(cat "${RUNNER_WORKSPACE}/../../.service")
echo "GitHub self-hosted runner service: ${RUNNER_SERVICE}"

if [[ -n "${RUNNER_SERVICE}" ]]; then
  echo "The self-hosted runner has encountered an irrecoverable error and will be shutdown"
  # and stop it to prevent the runner from receiving new jobs
  sudo systemctl stop "${RUNNER_SERVICE}"
fi
