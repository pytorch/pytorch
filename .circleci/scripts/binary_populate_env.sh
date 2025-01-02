#!/bin/bash
set -eux -o pipefail
export TZ=UTC

tagged_version() {
  GIT_DIR="${workdir}/pytorch/.git"
  GIT_DESCRIBE="git --git-dir ${GIT_DIR} describe --tags --match v[0-9]*.[0-9]*.[0-9]*"
  if [[ ! -d "${GIT_DIR}" ]]; then
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
    export DOCKER_IMAGE="pytorch/manylinux:cpu"
  else
    export DOCKER_IMAGE="pytorch/manylinux-builder:${DESIRED_CUDA:2}"
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
BASE_BUILD_VERSION="$(cat ${PYTORCH_ROOT}/version.txt|cut -da -f1).dev${DATE}"

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

export PYTORCH_BUILD_NUMBER=1

# Set triton version as part of PYTORCH_EXTRA_INSTALL_REQUIREMENTS
TRITON_VERSION=$(cat $PYTORCH_ROOT/.ci/docker/triton_version.txt)

# Here PYTORCH_EXTRA_INSTALL_REQUIREMENTS is already set for the all the wheel builds hence append TRITON_CONSTRAINT
TRITON_CONSTRAINT="platform_system == 'Linux' and platform_machine == 'x86_64' and python_version != '3.13t'"
if [[ "$PACKAGE_TYPE" =~ .*wheel.* &&  -n "${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}" ]]; then
  TRITON_REQUIREMENT="triton==${TRITON_VERSION}; ${TRITON_CONSTRAINT}"
  # Only linux Python <= 3.13, not 3.13t are supported wheels for triton
  if [[ -n "$PYTORCH_BUILD_VERSION" && "$PYTORCH_BUILD_VERSION" =~ .*dev.* ]]; then
      TRITON_SHORTHASH=$(cut -c1-8 $PYTORCH_ROOT/.ci/docker/ci_commit_pins/triton.txt)
      TRITON_REQUIREMENT="pytorch-triton==${TRITON_VERSION}+git${TRITON_SHORTHASH}; ${TRITON_CONSTRAINT}"
  fi
  export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS} | ${TRITON_REQUIREMENT}"
fi

# Set triton via PYTORCH_EXTRA_INSTALL_REQUIREMENTS for triton rocm package
if [[ "$PACKAGE_TYPE" =~ .*wheel.* && -n "$PYTORCH_BUILD_VERSION" && "$PYTORCH_BUILD_VERSION" =~ .*rocm.* && $(uname) == "Linux" ]]; then
    TRITON_REQUIREMENT="pytorch-triton-rocm==${TRITON_VERSION}; ${TRITON_CONSTRAINT}"
    if [[ -n "$PYTORCH_BUILD_VERSION" && "$PYTORCH_BUILD_VERSION" =~ .*dev.* ]]; then
        TRITON_SHORTHASH=$(cut -c1-8 $PYTORCH_ROOT/.ci/docker/ci_commit_pins/triton.txt)
        TRITON_REQUIREMENT="pytorch-triton-rocm==${TRITON_VERSION}+git${TRITON_SHORTHASH}; ${TRITON_CONSTRAINT}"
    fi
    if [[ -z "${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}" ]]; then
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${TRITON_REQUIREMENT}"
    else
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS} | ${TRITON_REQUIREMENT}"
    fi
fi

# Set triton via PYTORCH_EXTRA_INSTALL_REQUIREMENTS for triton xpu package
if [[ "$PACKAGE_TYPE" =~ .*wheel.* && -n "$PYTORCH_BUILD_VERSION" && "$PYTORCH_BUILD_VERSION" =~ .*xpu.* && $(uname) == "Linux" ]]; then
    TRITON_REQUIREMENT="pytorch-triton-xpu==${TRITON_VERSION}; ${TRITON_CONSTRAINT}"
    if [[ -n "$PYTORCH_BUILD_VERSION" && "$PYTORCH_BUILD_VERSION" =~ .*dev.* ]]; then
        TRITON_SHORTHASH=$(cut -c1-8 $PYTORCH_ROOT/.ci/docker/ci_commit_pins/triton-xpu.txt)
        TRITON_REQUIREMENT="pytorch-triton-xpu==${TRITON_VERSION}+git${TRITON_SHORTHASH}; ${TRITON_CONSTRAINT}"
    fi
    if [[ -z "${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}" ]]; then
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${TRITON_REQUIREMENT}"
    else
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS} | ${TRITON_REQUIREMENT}"
    fi
fi

USE_GLOO_WITH_OPENSSL="ON"
if [[ "$GPU_ARCH_TYPE" =~ .*aarch64.* ]]; then
  USE_GLOO_WITH_OPENSSL="OFF"
  USE_GOLD_LINKER="OFF"
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
export USE_SPLIT_BUILD="${USE_SPLIT_BUILD:-}"
if [[ "${OSTYPE}" == "msys" ]]; then
  export LIBTORCH_CONFIG="${LIBTORCH_CONFIG:-}"
  if [[ "${LIBTORCH_CONFIG:-}" == 'debug' ]]; then
    export DEBUG=1
  fi
  export DESIRED_DEVTOOLSET=""
else
  export DESIRED_DEVTOOLSET="${DESIRED_DEVTOOLSET:-}"
fi

export DATE="$DATE"
export NIGHTLIES_DATE_PREAMBLE=1.14.0.dev
export PYTORCH_BUILD_VERSION="$PYTORCH_BUILD_VERSION"
export PYTORCH_BUILD_NUMBER="$PYTORCH_BUILD_NUMBER"
export OVERRIDE_PACKAGE_VERSION="$PYTORCH_BUILD_VERSION"
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-}"

# TODO: We don't need this anymore IIUC
export TORCH_PACKAGE_NAME='torch'
export TORCH_CONDA_BUILD_FOLDER='pytorch-nightly'
export ANACONDA_USER='pytorch'

export USE_FBGEMM=1
export PIP_UPLOAD_FOLDER="$PIP_UPLOAD_FOLDER"
export DOCKER_IMAGE="$DOCKER_IMAGE"


export USE_GOLD_LINKER="${USE_GOLD_LINKER}"
export USE_GLOO_WITH_OPENSSL="${USE_GLOO_WITH_OPENSSL}"
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

echo 'retry () {' >> "$envfile"
echo '    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)' >> "$envfile"
echo '}' >> "$envfile"
echo 'export -f retry' >> "$envfile"

cat "$envfile"
