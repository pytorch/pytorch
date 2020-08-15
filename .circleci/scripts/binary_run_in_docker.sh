#!/bin/bash

# This section is used in the binary_test and smoke_test jobs. It expects
# 'binary_populate_env' to have populated /home/circleci/project/env and it
# expects another section to populate /home/circleci/project/ci_test_script.sh
# with the code to run in the docker

# Expect all needed environment variables to be written to this file
source /home/circleci/project/env
echo "Running the following code in Docker"
cat /home/circleci/project/ci_test_script.sh
echo
echo
set -eux -o pipefail

# Expect actual code to be written to this file
chmod +x /home/circleci/project/ci_test_script.sh

VOLUME_MOUNTS="-v /home/circleci/project/:/circleci_stuff -v /home/circleci/project/final_pkgs:/final_pkgs -v ${PYTORCH_ROOT}:/pytorch -v ${BUILDER_ROOT}:/builder"
# Run the docker
if [ -n "${USE_CUDA_DOCKER_RUNTIME:-}" ]; then
  export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --runtime=nvidia ${VOLUME_MOUNTS} -t -d "${DOCKER_IMAGE}")
else
  export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ${VOLUME_MOUNTS} -t -d "${DOCKER_IMAGE}")
fi

# Execute the test script that was populated by an earlier section
export COMMAND='((echo "source /circleci_stuff/env && /circleci_stuff/ci_test_script.sh") | docker exec -i "$id" bash) 2>&1'
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts
