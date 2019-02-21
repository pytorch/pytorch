#!/bin/bash -xe

# TODO: merge this into Caffe2 test.sh
cat >/home/circleci/project/ci_test_script.sh <<EOL
# =================== The following code will be executed inside Docker container ===================
set -ex

export BUILD_ENVIRONMENT="$BUILD_ENVIRONMENT"

# libdc1394 (dependency of OpenCV) expects /dev/raw1394 to exist...
sudo ln /dev/null /dev/raw1394

# conda must be added to the path for Anaconda builds (this location must be
# the same as that in install_anaconda.sh used to build the docker image)
if [[ "${BUILD_ENVIRONMENT}" == conda* ]]; then
  export PATH=/opt/conda/bin:$PATH
fi

# Upgrade SSL module to avoid old SSL warnings
pip -q install --user --upgrade pyOpenSSL ndg-httpsclient pyasn1

pip -q install --user -b /tmp/pip_install_onnx "file:///var/lib/jenkins/workspace/third_party/onnx#egg=onnx"

# Build
./.jenkins/caffe2/test.sh

# Remove benign core dumps.
# These are tests for signal handling (including SIGABRT).
rm -f ./crash/core.fatal_signal_as.*
rm -f ./crash/core.logging_test.*
# =================== The above code will be executed inside Docker container ===================
EOL
chmod +x /home/circleci/project/ci_test_script.sh

if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-cmake-${CIRCLE_SHA1}
else
  export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
fi
echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}
docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null
if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  export id=$(docker run --runtime=nvidia -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
else
  export id=$(docker run -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
fi
docker cp /home/circleci/project/. "$id:/var/lib/jenkins/workspace"

export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && ./ci_test_script.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

