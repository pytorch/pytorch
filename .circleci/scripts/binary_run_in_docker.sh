#!/bin/bash

# This section is used in the binary_test and smoke_test jobs. It expects
# 'binary_populate_env' to have populated /home/circleci/project/env and it
# expects another section to populate /home/circleci/project/ci_test_script.sh
# with the code to run in the docker

# Expect all needed environment variables to be written to this file
source /home/circleci/project/env
echo "Running the following code in Docker"
cat /home/circleci/project/ci_test_script.sh
set -ex

# Expect actual code to be written to this file
chmod +x /home/circleci/project/ci_test_script.sh

# Run the docker
if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  export id=$(docker run --runtime=nvidia -t -d "${DOCKER_IMAGE}")
else
  export id=$(docker run -t -d "${DOCKER_IMAGE}")
fi

# Copy the envfile and script with all the code to run into the docker.
docker cp /home/circleci/project/. "$id:/circleci_stuff"

# Copy built packages into the docker to test. This should only exist on the
# binary test jobs. The package should've been created from a binary build job,
# whhich persisted the package to a CircleCI workspace, which this job then
# copies into a GPU enabled docker for testing
if [[ -d "/home/circleci/project/final_pkgs" ]]; then
  docker cp /home/circleci/project/final_pkgs "$id:/final_pkgs"
fi

# Copy the needed repos into the docker. These do not exist in the smoke test
# jobs, since the smoke test jobs do not need the Pytorch source code.
if [[ -d "$PYTORCH_ROOT" ]]; then
  docker cp "$PYTORCH_ROOT" "$id:/pytorch"
fi
if [[ -d "$BUILDER_ROOT" ]]; then
  docker cp "$BUILDER ROOT" "$id:/builder"
fi

# Execute the test script that was populated by an earlier section
export COMMAND='((echo "source /circleci_stuff/env && /circleci_stuff/ci_test_script.sh") | docker exec -i "$id" bash) 2>&1'
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts
