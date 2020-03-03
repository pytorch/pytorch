#!/bin/bash
set -eux -o pipefail
export TZ=UTC

# We need to write an envfile to persist these variables to following
# steps, but the location of the envfile depends on the circleci executor
if [[ "$(uname)" == Darwin ]]; then
  # macos executor (builds and tests)
  workdir="/Users/distiller/project"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  workdir="/home/circleci/project"
else
  # docker executor (binary builds)
  workdir="/"
fi
envfile="$workdir/env"
touch "$envfile"
chmod +x "$envfile"

# Parse the BUILD_ENVIRONMENT to package type, python, and cuda
configs=($BUILD_ENVIRONMENT)
export PACKAGE_TYPE="${configs[0]}"
export DESIRED_PYTHON="${configs[1]}"
export DESIRED_CUDA="${configs[2]}"
export DESIRED_DEVTOOLSET="${configs[3]:-}"
if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
  export BUILD_PYTHONLESS=1
fi

# Pick docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-}
if [[ -z "$DOCKER_IMAGE" ]]; then
  if [[ "$PACKAGE_TYPE" == conda ]]; then
    export DOCKER_IMAGE="pytorch/conda-cuda"
  elif [[ "$DESIRED_CUDA" == cpu ]]; then
    export DOCKER_IMAGE="pytorch/manylinux-cuda100"
  else
    export DOCKER_IMAGE="pytorch/manylinux-cuda${DESIRED_CUDA:2}"
  fi
fi

# Default to nightly, since that's where this normally uploads to
PIP_UPLOAD_FOLDER='nightly/'
# We put this here so that OVERRIDE_PACKAGE_VERSION below can read from it
export DATE="$(date -u +%Y%m%d)"
#TODO: We should be pulling semver version from the base version.txt
BASE_BUILD_VERSION="1.5.0.dev$DATE"
# Change BASE_BUILD_VERSION to git tag when on a git tag
if git describe --tags --exact >/dev/null 2>/dev/null; then
  # Switch upload folder to 'test/' if we are on a tag
  PIP_UPLOAD_FOLDER='test/'
  # Grab git tag, remove prefixed v and remove everything after -
  # Used to clean up tags that are for release candidates like v1.5.0-rc1
  # Turns tag v1.5.0-rc1 -> v1.5.0
  BASE_BUILD_VERSION="$(git describe --tags | sed -e 's/^v//' -e 's/-.*$//')"
fi
if [[ "$(uname)" == 'Darwin' ]] || [[ "$DESIRED_CUDA" == "cu101" ]] || [[ "$PACKAGE_TYPE" == conda ]]; then
  export PYTORCH_BUILD_VERSION="${BASE_BUILD_VERSION}"
else
  export PYTORCH_BUILD_VERSION="${BASE_BUILD_VERSION}+$DESIRED_CUDA"
fi
export PYTORCH_BUILD_NUMBER=1


JAVA_HOME=
BUILD_JNI=OFF
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  POSSIBLE_JAVA_HOMES=()
  POSSIBLE_JAVA_HOMES+=(/usr/local)
  POSSIBLE_JAVA_HOMES+=(/usr/lib/jvm/java-8-openjdk-amd64)
  POSSIBLE_JAVA_HOMES+=(/Library/Java/JavaVirtualMachines/*.jdk/Contents/Home)
  for JH in "${POSSIBLE_JAVA_HOMES[@]}" ; do
    if [[ -e "$JH/include/jni.h" ]] ; then
      echo "Found jni.h under $JH"
      JAVA_HOME="$JH"
      BUILD_JNI=ON
      break
    fi
  done
  if [ -z "$JAVA_HOME" ]; then
    echo "Did not find jni.h"
  fi
fi

cat >>"$envfile" <<EOL
# =================== The following code will be executed inside Docker container ===================
export TZ=UTC
echo "Running on $(uname -a) at $(date)"

export PACKAGE_TYPE="$PACKAGE_TYPE"
export DESIRED_PYTHON="$DESIRED_PYTHON"
export DESIRED_CUDA="$DESIRED_CUDA"
export LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-}"
export BUILD_PYTHONLESS="${BUILD_PYTHONLESS:-}"
export DESIRED_DEVTOOLSET="$DESIRED_DEVTOOLSET"

export DATE="$DATE"
export NIGHTLIES_DATE_PREAMBLE=1.5.0.dev
export PYTORCH_BUILD_VERSION="$PYTORCH_BUILD_VERSION"
export PYTORCH_BUILD_NUMBER="$PYTORCH_BUILD_NUMBER"
export OVERRIDE_PACKAGE_VERSION="$PYTORCH_BUILD_VERSION"

# TODO: We don't need this anymore IIUC
export TORCH_PACKAGE_NAME='torch'
export TORCH_CONDA_BUILD_FOLDER='pytorch-nightly'

export USE_FBGEMM=1
export JAVA_HOME=$JAVA_HOME
export BUILD_JNI=$BUILD_JNI
export PIP_UPLOAD_FOLDER="$PIP_UPLOAD_FOLDER"
export DOCKER_IMAGE="$DOCKER_IMAGE"

export workdir="$workdir"
export MAC_PACKAGE_WORK_DIR="$workdir"
export PYTORCH_ROOT="$workdir/pytorch"
export BUILDER_ROOT="$workdir/builder"
export MINICONDA_ROOT="$workdir/miniconda"
export PYTORCH_FINAL_PACKAGE_DIR="$workdir/final_pkgs"

export CIRCLE_TAG="${CIRCLE_TAG:-}"
export CIRCLE_SHA1="$CIRCLE_SHA1"
export CIRCLE_PR_NUMBER="${CIRCLE_PR_NUMBER:-}"
export CIRCLE_BRANCH="$CIRCLE_BRANCH"
# =================== The above code will be executed inside Docker container ===================
EOL

echo 'retry () {' >> "$envfile"
echo '    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)' >> "$envfile"
echo '}' >> "$envfile"
echo 'export -f retry' >> "$envfile"

cat "$envfile"
