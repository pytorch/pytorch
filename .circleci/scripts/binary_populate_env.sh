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
if [[ "$PACKAGE_TYPE" == conda ]]; then
  export DOCKER_IMAGE="soumith/conda-cuda"
elif [[ "$DESIRED_CUDA" == cpu ]]; then
  export DOCKER_IMAGE="soumith/manylinux-cuda100"
else
  export DOCKER_IMAGE="soumith/manylinux-cuda${DESIRED_CUDA:2}"
fi

# Upload to parallel folder for gcc abis
# All nightlies used to be devtoolset3, then devtoolset7 was added as a build
# option, so the upload was redirected to nightly/devtoolset7 to avoid
# conflicts with other binaries (there shouldn't be any conflicts). Now we are
# making devtoolset7 the default.
if [[ "$DESIRED_DEVTOOLSET" == 'devtoolset7' || "$(uname)" == 'Darwin' ]]; then
  export PIP_UPLOAD_FOLDER='nightly/'
else
  # On linux machines, this shouldn't actually be called anymore. This is just
  # here for extra safety.
  export PIP_UPLOAD_FOLDER='nightly/devtoolset3/'
fi

# We put this here so that OVERRIDE_PACKAGE_VERSION below can read from it
export DATE="$(date -u +%Y%m%d)"
if [[ "$(uname)" == 'Darwin' ]] || [[ "$DESIRED_CUDA" == "cu100" ]] || [[ "$PACKAGE_TYPE" == conda ]]; then
  export PYTORCH_BUILD_VERSION="1.3.0.dev$DATE"
else
  export PYTORCH_BUILD_VERSION="1.3.0.dev$DATE+$DESIRED_CUDA"
fi
export PYTORCH_BUILD_NUMBER=1

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
export NIGHTLIES_DATE_PREAMBLE=1.3.0.dev
export PYTORCH_BUILD_VERSION="$PYTORCH_BUILD_VERSION"
export PYTORCH_BUILD_NUMBER="$PYTORCH_BUILD_NUMBER"
export OVERRIDE_PACKAGE_VERSION="$PYTORCH_BUILD_VERSION"

# TODO: We don't need this anymore IIUC
export TORCH_PACKAGE_NAME='torch'
export TORCH_CONDA_BUILD_FOLDER='pytorch-nightly'

export USE_FBGEMM=1
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
