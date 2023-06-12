#!/bin/bash

set +e
set -x

# Get the service name
RUNNER_SERVICE=$(cat "${RUNNER_WORKSPACE}/../../.service")
echo "GitHub self-hosted runner service: ${RUNNER_SERVICE}"

if [[ -n "${RUNNER_SERVICE}" ]]; then
  echo "The self-hosted runner has encountered an unrecoverable error and will be shutdown"

  pushd "${RUNNER_WORKSPACE}/../../"
  # Stop it to prevent the runner from receiving new jobs
  sudo ./svc.sh stop
  # then uninstall the service
  sudo ./svc.sh uninstall
  # Finally, shutting down the runner completely
  sudo shutdown -P now
  # NB: In my test, cleaning up and shutting down the runner this way would already
  # remove the runner from the list of registered runners. Calling config.sh remove
  # seems redundant as it would require an org token to use, which I don't want to
  # add as yet another secret to the CI if there is no need
fi
