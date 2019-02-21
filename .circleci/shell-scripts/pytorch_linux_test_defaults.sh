#!/bin/bash -xe

export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}
docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null
if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  export id=$(docker run --runtime=nvidia -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
else
  export id=$(docker run -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
fi
if [ -n "${MULTI_GPU}" ]; then
  export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/multigpu-test.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
else
  export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/test.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
fi
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts
