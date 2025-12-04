#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/../pytorch/common_utils.sh"

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/test"
pytest_reports_dir="${TEST_DIR}/test-reports/python"

# Figure out which Python to use
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON=$(which "python${BASH_REMATCH[1]}")
fi

if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
    # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
    unset HIP_PLATFORM
fi

mkdir -p "$pytest_reports_dir" || true

##########################################
# copied from .ci/pytorch/common_utils.sh
##########################################

function get_pinned_commit() {
  cat .github/ci_commit_pins/"${1}".txt
}

function pip_install_whl() {
  # This is used to install PyTorch and other build artifacts wheel locally
  # without using any network connection

  # Convert the input arguments into an array
  local args=("$@")

  # Check if the first argument contains multiple paths separated by spaces
  if [[ "${args[0]}" == *" "* ]]; then
    # Split the string by spaces into an array
    IFS=' ' read -r -a paths <<< "${args[0]}"
    # Loop through each path and install individually
    for path in "${paths[@]}"; do
      echo "Installing $path"
      python3 -mpip install --no-index --no-deps "$path"
    done
  else
    # Loop through each argument and install individually
    for path in "${args[@]}"; do
      echo "Installing $path"
      python3 -mpip install --no-index --no-deps "$path"
    done
  fi
}

function pip_build_and_install() {
  local build_target=$1
  local wheel_dir=$2

  local found_whl=0
  for file in "${wheel_dir}"/*.whl
  do
    if [[ -f "${file}" ]]; then
      found_whl=1
      break
    fi
  done

  # Build the wheel if it doesn't exist
  if [ "${found_whl}" == "0" ]; then
    python3 -m pip wheel \
      --no-build-isolation \
      --no-deps \
      -w "${wheel_dir}" \
      "${build_target}"
  fi

  for file in "${wheel_dir}"/*.whl
  do
    pip_install_whl "${file}"
  done
}

function install_torchvision() {
  local orig_preload
  local commit
  commit=$(get_pinned_commit vision)
  orig_preload=${LD_PRELOAD}
  if [ -n "${LD_PRELOAD}" ]; then
    # Silence dlerror to work-around glibc ASAN bug, see https://sourceware.org/bugzilla/show_bug.cgi?id=27653#c9
    echo 'char* dlerror(void) { return "";}'|gcc -fpic -shared -o "${HOME}/dlerror.so" -x c -
    LD_PRELOAD=${orig_preload}:${HOME}/dlerror.so
  fi

  if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
    # Not sure if both are needed, but why not
    export FORCE_CUDA=1
    export WITH_CUDA=1
  fi
  pip_build_and_install "git+https://github.com/pytorch/vision.git@${commit}" dist/vision

  if [ -n "${LD_PRELOAD}" ]; then
    LD_PRELOAD=${orig_preload}
  fi
}
