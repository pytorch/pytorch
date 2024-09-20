#!/bin/bash
#
##############################################################################
# Example command to build Caffe2
##############################################################################
#

set -ex

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

CMAKE_ARGS=()

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

# Use ccache if available (this path is where Homebrew installs ccache symlinks)
if [ "$(uname)" == 'Darwin' ]; then
  if [ -n "${CCACHE_WRAPPER_PATH:-}"]; then
    CCACHE_WRAPPER_PATH=/usr/local/opt/ccache/libexec
  fi
  if [ -d "$CCACHE_WRAPPER_PATH" ]; then
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$CCACHE_WRAPPER_PATH/gcc")
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CCACHE_WRAPPER_PATH/g++")
  fi
fi

# Use special install script with Anaconda
if [ -n "${USE_ANACONDA}" ]; then
  export SKIP_CONDA_TESTS=1
  export CONDA_INSTALL_LOCALLY=1
  "${ROOT_DIR}/scripts/build_anaconda.sh" "$@"
else
  # Make sure that pyyaml is installed for the codegen of building Aten to work
  if [[ -n "$(python -c 'import yaml' 2>&1)" ]]; then
    echo "Installing pyyaml with pip at $(which pip)"
    pip install --user pyyaml
  fi

  # Make sure that typing is installed for the codegen of building Aten to work
  if [[ -n "$(python -c 'import typing' 2>&1)" ]]; then
    echo "Installing typing with pip at $(which pip)"
    pip install --user typing
  fi

  # Build protobuf compiler from third_party if configured to do so
  if [ -n "${USE_HOST_PROTOC:-}" ]; then
    echo "USE_HOST_PROTOC is set; building protoc before building Caffe2..."
    "$CAFFE2_ROOT/scripts/build_host_protoc.sh"
    CUSTOM_PROTOC_EXECUTABLE="$CAFFE2_ROOT/build_host_protoc/bin/protoc"
    echo "Built protoc $("$CUSTOM_PROTOC_EXECUTABLE" --version)"
    CMAKE_ARGS+=("-DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$CUSTOM_PROTOC_EXECUTABLE")
  fi

  # We are going to build the target into build.
  BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}
  mkdir -p "$BUILD_ROOT"
  cd "$BUILD_ROOT"
  echo "Building Caffe2 in: $BUILD_ROOT"

  cmake "$CAFFE2_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        "${CMAKE_ARGS[@]}" \
        "$@"

  # Determine the number of CPUs to build with.
  # If the `CAFFE_MAKE_NCPUS` variable is not specified, use them all.
  if [ -n "${MAX_JOBS}" ]; then
      CAFFE_MAKE_NCPUS="$MAX_JOBS"
  elif [ -n "${CAFFE_MAKE_NCPUS}" ]; then
      CAFFE_MAKE_NCPUS="$CAFFE_MAKE_NCPUS"
  elif [ "$(uname)" == 'Darwin' ]; then
      CAFFE_MAKE_NCPUS="$(sysctl -n hw.ncpu)"
  else
      CAFFE_MAKE_NCPUS="$(nproc)"
  fi

  # Now, actually build the target.
  cmake --build . -- "-j$CAFFE_MAKE_NCPUS"
fi
