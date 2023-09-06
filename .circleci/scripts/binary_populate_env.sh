#!/bin/bash
set -eux -o pipefail
export TZ=UTC

tagged_version() {
  # Grabs version from either the env variable CIRCLE_TAG
  # or the pytorch git described version
  if [[ "$OSTYPE" == "msys" &&  -z "${GITHUB_ACTIONS:-}" ]]; then
    GIT_DIR="${workdir}/p/.git"
  else
    GIT_DIR="${workdir}/pytorch/.git"
  fi
  GIT_DESCRIBE="git --git-dir ${GIT_DIR} describe --tags --match v[0-9]*.[0-9]*.[0-9]*"
  if [[ -n "${CIRCLE_TAG:-}" ]]; then
    echo "${CIRCLE_TAG}"
  elif [[ ! -d "${GIT_DIR}" ]]; then
    echo "Abort, abort! Git dir ${GIT_DIR} does not exists!"
    kill $$
  elif ${GIT_DESCRIBE} --exact >/dev/null; then
    ${GIT_DESCRIBE}
  else
    return 1
  fi
}

envfile=${BINARY_ENV_FILE:-/tmp/env}
if [[ -n "${PYTORCH_ROOT}"  ]]; then
  workdir=$(dirname "${PYTORCH_ROOT}")
else
  # docker executor (binary builds)
  workdir="/"
fi

if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
  export BUILD_PYTHONLESS=1
fi

# Pick docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-}
if [[ -z "$DOCKER_IMAGE" ]]; then
  if [[ "$PACKAGE_TYPE" == conda ]]; then
    export DOCKER_IMAGE="pytorch/conda-cuda"
  elif [[ "$DESIRED_CUDA" == cpu ]]; then
    export DOCKER_IMAGE="pytorch/manylinux-cpu"
  else
    export DOCKER_IMAGE="pytorch/manylinux-cuda${DESIRED_CUDA:2}"
  fi
fi

USE_GOLD_LINKER="OFF"
# GOLD linker can not be used if CUPTI is statically linked into PyTorch, see https://github.com/pytorch/pytorch/issues/57744
if [[ ${DESIRED_CUDA} == "cpu" ]]; then
  USE_GOLD_LINKER="ON"
fi


# Default to nightly, since that's where this normally uploads to
PIP_UPLOAD_FOLDER='nightly/'
# We put this here so that OVERRIDE_PACKAGE_VERSION below can read from it
export DATE="$(date -u +%Y%m%d)"
#TODO: We should be pulling semver version from the base version.txt
BASE_BUILD_VERSION="2.1.0.dev$DATE"
# Change BASE_BUILD_VERSION to git tag when on a git tag
# Use 'git -C' to make doubly sure we're in the correct directory for checking
# the git tag
if tagged_version >/dev/null; then
  # Switch upload folder to 'test/' if we are on a tag
  PIP_UPLOAD_FOLDER='test/'
  # Grab git tag, remove prefixed v and remove everything after -
  # Used to clean up tags that are for release candidates like v1.6.0-rc1
  # Turns tag v1.6.0-rc1 -> v1.6.0
  BASE_BUILD_VERSION="$(tagged_version | sed -e 's/^v//' -e 's/-.*$//')"
fi
if [[ "$(uname)" == 'Darwin' ]] || [[ "$PACKAGE_TYPE" == conda ]]; then
  export PYTORCH_BUILD_VERSION="${BASE_BUILD_VERSION}"
else
  export PYTORCH_BUILD_VERSION="${BASE_BUILD_VERSION}+$DESIRED_CUDA"
fi

if [[ -n "${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}" ]]; then
  export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION}-with-pypi-cudnn"
fi

export PYTORCH_BUILD_NUMBER=1


JAVA_HOME=
BUILD_JNI=OFF
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  POSSIBLE_JAVA_HOMES=()
  POSSIBLE_JAVA_HOMES+=(/usr/local)
  POSSIBLE_JAVA_HOMES+=(/usr/lib/jvm/java-8-openjdk-amd64)
  POSSIBLE_JAVA_HOMES+=(/Library/Java/JavaVirtualMachines/*.jdk/Contents/Home)
  # Add the Windows-specific JNI path
  POSSIBLE_JAVA_HOMES+=("$PWD/pytorch/.circleci/windows-jni/")
  for JH in "${POSSIBLE_JAVA_HOMES[@]}" ; do
    if [[ -e "$JH/include/jni.h" ]] ; then
      # Skip if we're not on Windows but haven't found a JAVA_HOME
      if [[ "$JH" == "$PWD/pytorch/.circleci/windows-jni/" && "$OSTYPE" != "msys" ]] ; then
        break
      fi
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

cat >"$envfile" <<EOL
# =================== The following code will be executed inside Docker container ===================
export TZ=UTC
echo "Running on $(uname -a) at $(date)"

export PACKAGE_TYPE="$PACKAGE_TYPE"
export DESIRED_PYTHON="${DESIRED_PYTHON:-}"
export DESIRED_CUDA="$DESIRED_CUDA"
export LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-}"
export BUILD_PYTHONLESS="${BUILD_PYTHONLESS:-}"
if [[ "${OSTYPE}" == "msys" ]]; then
  export LIBTORCH_CONFIG="${LIBTORCH_CONFIG:-}"
  if [[ "${LIBTORCH_CONFIG:-}" == 'debug' ]]; then
    export DEBUG=1
  fi
  export DESIRED_DEVTOOLSET=""
else
  export DESIRED_DEVTOOLSET="${DESIRED_DEVTOOLSET:-}"
fi
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}"
export DATE="$DATE"
export NIGHTLIES_DATE_PREAMBLE=1.14.0.dev
export PYTORCH_BUILD_VERSION="$PYTORCH_BUILD_VERSION"
export PYTORCH_BUILD_NUMBER="$PYTORCH_BUILD_NUMBER"
export OVERRIDE_PACKAGE_VERSION="$PYTORCH_BUILD_VERSION"

# TODO: We don't need this anymore IIUC
export TORCH_PACKAGE_NAME='torch'
export TORCH_CONDA_BUILD_FOLDER='pytorch-nightly'
export ANACONDA_USER='pytorch'

export USE_FBGEMM=1
export JAVA_HOME=$JAVA_HOME
export BUILD_JNI=$BUILD_JNI
export PIP_UPLOAD_FOLDER="$PIP_UPLOAD_FOLDER"
export DOCKER_IMAGE="$DOCKER_IMAGE"


export USE_GOLD_LINKER="${USE_GOLD_LINKER}"
export USE_GLOO_WITH_OPENSSL="ON"
# =================== The above code will be executed inside Docker container ===================
EOL

# nproc doesn't exist on darwin
if [[ "$(uname)" != Darwin ]]; then
  # This was lowered from 18 to 12 to avoid OOMs when compiling FlashAttentionV2
  MEMORY_LIMIT_MAX_JOBS=12
  NUM_CPUS=$(( $(nproc) - 2 ))

  # Defaults here for **binary** linux builds so they can be changed in one place
  export MAX_JOBS=${MAX_JOBS:-$(( ${NUM_CPUS} > ${MEMORY_LIMIT_MAX_JOBS} ? ${MEMORY_LIMIT_MAX_JOBS} : ${NUM_CPUS} ))}

  cat >>"$envfile" <<EOL
  export MAX_JOBS="${MAX_JOBS}"
EOL
fi

if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  cat >>"$envfile" <<EOL
  export workdir="$workdir"
  export MAC_PACKAGE_WORK_DIR="$workdir"
  if [[ "$OSTYPE" == "msys" ]]; then
    export PYTORCH_ROOT="$workdir/p"
    export BUILDER_ROOT="$workdir/b"
  else
    export PYTORCH_ROOT="$workdir/pytorch"
    export BUILDER_ROOT="$workdir/builder"
  fi
  export MINICONDA_ROOT="$workdir/miniconda"
  export PYTORCH_FINAL_PACKAGE_DIR="$workdir/final_pkgs"

  export CIRCLE_TAG="${CIRCLE_TAG:-}"
  export CIRCLE_SHA1="$CIRCLE_SHA1"
  export CIRCLE_PR_NUMBER="${CIRCLE_PR_NUMBER:-}"
  export CIRCLE_BRANCH="$CIRCLE_BRANCH"
  export CIRCLE_WORKFLOW_ID="$CIRCLE_WORKFLOW_ID"
EOL
fi

echo 'retry () {' >> "$envfile"
echo '    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)' >> "$envfile"
echo '}' >> "$envfile"
echo 'export -f retry' >> "$envfile"

cat "$envfile"
