#!/bin/bash -xe

cat >/home/circleci/project/ci_build_script.sh <<EOL
# =================== The following code will be executed inside Docker container ===================
set -ex
export BUILD_ENVIRONMENT="$BUILD_ENVIRONMENT"

# Reinitialize submodules
git submodule sync && git submodule update -q --init --recursive

# conda must be added to the path for Anaconda builds (this location must be
# the same as that in install_anaconda.sh used to build the docker image)
if [[ "${BUILD_ENVIRONMENT}" == conda* ]]; then
  export PATH=/opt/conda/bin:$PATH
  sudo chown -R jenkins:jenkins '/opt/conda'
fi

# Build
./.jenkins/caffe2/build.sh

# Show sccache stats if it is running
if pgrep sccache > /dev/null; then
  sccache --show-stats
fi
# =================== The above code will be executed inside Docker container ===================
EOL
chmod +x /home/circleci/project/ci_build_script.sh

echo "DOCKER_IMAGE: "${DOCKER_IMAGE}
docker pull ${DOCKER_IMAGE} >/dev/null
export id=$(docker run -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})
docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace

export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && ./ci_build_script.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

# Push intermediate Docker image for next phase to use
if [ -z "${BUILD_ONLY}" ]; then
  if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
    export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-cmake-${CIRCLE_SHA1}
  else
    export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
  fi
  docker commit "$id" ${COMMIT_DOCKER_IMAGE}
  docker push ${COMMIT_DOCKER_IMAGE}
fi
