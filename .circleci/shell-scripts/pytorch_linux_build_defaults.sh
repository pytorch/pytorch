#!/bin/bash -xe

# Pull Docker image and run build
echo "DOCKER_IMAGE: "${DOCKER_IMAGE}
docker pull ${DOCKER_IMAGE} >/dev/null
export id=$(docker run -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})

git submodule sync && git submodule update -q --init

docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace

export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

# Push intermediate Docker image for next phase to use
if [ -z "${BUILD_ONLY}" ]; then
  export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
  docker commit "$id" ${COMMIT_DOCKER_IMAGE}
  docker push ${COMMIT_DOCKER_IMAGE}
fi
