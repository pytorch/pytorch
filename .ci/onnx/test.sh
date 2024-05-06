#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Workaround for dind-rootless userid mapping (https://github.com/pytorch/ci-infra/issues/96)
WORKSPACE_ORIGINAL_OWNER_ID=$(stat -c '%u' "/var/lib/jenkins/workspace")
cleanup_workspace() {
  echo "sudo may print the following warning message that can be ignored. The chown command will still run."
  echo "    sudo: setrlimit(RLIMIT_STACK): Operation not permitted"
  echo "For more details refer to https://github.com/sudo-project/sudo/issues/42"
  sudo chown -R "$WORKSPACE_ORIGINAL_OWNER_ID" /var/lib/jenkins/workspace
}
# Disable shellcheck SC2064 as we want to parse the original owner immediately.
# shellcheck disable=SC2064
trap_add cleanup_workspace EXIT
sudo chown -R jenkins /var/lib/jenkins/workspace
git config --global --add safe.directory /var/lib/jenkins/workspace

if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  # TODO: This can be removed later once vision is also part of the Docker image
  pip install -q --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$(cat .github/ci_commit_pins/vision.txt)"
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  # NB: ONNX test is fast (~15m) so it's ok to retry it few more times to avoid any flaky issue, we
  # need to bring this to the standard PyTorch run_test eventually. The issue will be tracked in
  # https://github.com/pytorch/pytorch/issues/98626
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
