#!/bin/bash

# Common util **functions** that can be sourced in other scripts.

# note: printf is used instead of echo to avoid backslash
# processing and to properly handle values that begin with a '-'.

log() { printf '%s\n' "$*"; }
error() { log "ERROR: $*" >&2; }
fatal() { error "$@"; exit 1; }

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

# compositional trap taken from https://stackoverflow.com/a/7287873/23845
# appends a command to a trap
#
# - 1st arg:  code to add
# - remaining args:  names of traps to modify
#
trap_add() {
    trap_add_cmd=$1; shift || fatal "${FUNCNAME[0]} usage error"
    for trap_add_name in "$@"; do
        trap -- "$(
            # helper fn to get existing trap command from output
            # of trap -p
            extract_trap_cmd() { printf '%s\n' "$3"; }
            # print existing trap command with newline
            eval "extract_trap_cmd $(trap -p "${trap_add_name}")"
            # print the new trap command
            printf '%s\n' "${trap_add_cmd}"
        )" "${trap_add_name}" \
            || fatal "unable to add to trap ${trap_add_name}"
    done
}
# set the trace attribute for the above function.  this is
# required to modify DEBUG or RETURN traps because functions don't
# inherit them unless the trace attribute is set
declare -f -t trap_add

function assert_git_not_dirty() {
    # TODO: we should add an option to `build_amd.py` that reverts the repo to
    #       an unmodified state.
    if [[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *xla* ]] && [[ "$BUILD_ENVIRONMENT" != *aarch64* ]] ; then
        git_status=$(git status --porcelain | grep -v '?? third_party' || true)
        if [[ $git_status ]]; then
            echo "Build left local git repository checkout dirty"
            echo "git status --porcelain:"
            echo "${git_status}"
            exit 1
        fi
    fi
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

function pip_install() {
  # retry 3 times
  pip_install_pkg="python3 -m pip install --progress-bar off"
  ${pip_install_pkg} "$@" || \
    ${pip_install_pkg} "$@" || \
    ${pip_install_pkg} "$@"
}

function pip_uninstall() {
  # uninstall 2 times
  pip3 uninstall -y "$@" || pip3 uninstall -y "$@"
}

function get_exit_code() {
  set +e
  "$@"
  retcode=$?
  set -e
  return $retcode
}

function get_bazel() {
  # Download and use the cross-platform, dependency-free Python
  # version of Bazelisk to fetch the platform specific version of
  # Bazel to use from .bazelversion.
  retry curl --location --output tools/bazel \
    https://raw.githubusercontent.com/bazelbuild/bazelisk/v1.23.0/bazelisk.py
  shasum --algorithm=1 --check \
    <(echo '01df9cf7f08dd80d83979ed0d0666a99349ae93c  tools/bazel')
  chmod u+x tools/bazel
}

function install_monkeytype {
  # Install MonkeyType
  pip_install MonkeyType
}


function get_pinned_commit() {
  cat .github/ci_commit_pins/"${1}".txt
}

function detect_cuda_arch() {
  if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
    if command -v nvidia-smi; then
      TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1)
    elif [[ "${TEST_CONFIG}" == *nogpu* ]]; then
      # There won't be nvidia-smi in nogpu tests, so just set TORCH_CUDA_ARCH_LIST to the default
      # minimum supported value here
      TORCH_CUDA_ARCH_LIST=8.0
    fi
    export TORCH_CUDA_ARCH_LIST
  fi
}

function install_torchaudio() {
  local commit
  commit=$(get_pinned_commit audio)
  pip_build_and_install "git+https://github.com/pytorch/audio.git@${commit}" dist/audio
}

function install_torchtext() {
  local data_commit
  local text_commit
  data_commit=$(get_pinned_commit data)
  text_commit=$(get_pinned_commit text)
  pip_build_and_install "git+https://github.com/pytorch/data.git@${data_commit}" dist/data
  pip_build_and_install "git+https://github.com/pytorch/text.git@${text_commit}" dist/text
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

function install_torchrec_and_fbgemm() {
  local torchrec_commit
  torchrec_commit=$(get_pinned_commit torchrec)
  local fbgemm_commit
  fbgemm_commit=$(get_pinned_commit fbgemm)
  if [[ "$BUILD_ENVIRONMENT" == *rocm* ]] ; then
    fbgemm_commit=$(get_pinned_commit fbgemm_rocm)
  fi
  pip_uninstall torchrec-nightly
  pip_uninstall fbgemm-gpu-nightly
  pip_install setuptools-git-versioning scikit-build pyre-extensions

  if [[ "$BUILD_ENVIRONMENT" == *rocm* ]] ; then
    # install torchrec first because it installs fbgemm nightly on top of rocm fbgemm
    pip_build_and_install "git+https://github.com/pytorch/torchrec.git@${torchrec_commit}" dist/torchrec
    pip_uninstall fbgemm-gpu-nightly

    # Set ROCM_HOME isn't available, use ROCM_PATH if set or /opt/rocm
    ROCM_HOME="${ROCM_HOME:-${ROCM_PATH:-/opt/rocm}}"

    # Find rocm_version.h header file for ROCm version extract
    rocm_version_h="${ROCM_HOME}/include/rocm-core/rocm_version.h"
    if [ ! -f "$rocm_version_h" ]; then
        rocm_version_h="${ROCM_HOME}/include/rocm_version.h"
    fi

    # Error out if rocm_version.h not found
    if [ ! -f "$rocm_version_h" ]; then
        echo "Error: rocm_version.h not found in expected locations." >&2
        exit 1
    fi

    # Extract major, minor and patch ROCm version numbers
    MAJOR_VERSION=$(grep 'ROCM_VERSION_MAJOR' "$rocm_version_h" | awk '{print $3}')
    MINOR_VERSION=$(grep 'ROCM_VERSION_MINOR' "$rocm_version_h" | awk '{print $3}')
    PATCH_VERSION=$(grep 'ROCM_VERSION_PATCH' "$rocm_version_h" | awk '{print $3}')
    ROCM_INT=$((MAJOR_VERSION * 10000 + MINOR_VERSION * 100 + PATCH_VERSION))
    echo "ROCm version: $ROCM_INT"
    export BUILD_ROCM_VERSION="$MAJOR_VERSION.$MINOR_VERSION"

    pip_install tabulate  # needed for newer fbgemm
    pip_install patchelf  # needed for rocm fbgemm

    local wheel_dir=dist/fbgemm_gpu
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
      git clone --recursive https://github.com/pytorch/fbgemm
      pushd fbgemm/fbgemm_gpu
      git checkout "${fbgemm_commit}" --recurse-submodules
      # until the fbgemm_commit includes the tbb patch
      patch <<'EOF'
--- a/FbgemmGpu.cmake
+++ b/FbgemmGpu.cmake
@@ -184,5 +184,6 @@ gpu_cpp_library(
     fbgemm_gpu_tbe_cache
     fbgemm_gpu_tbe_optimizers
     fbgemm_gpu_tbe_utils
+    tbb
   DESTINATION
     fbgemm_gpu)
EOF
      python setup.py bdist_wheel --build-variant=rocm
      popd

      # Save the wheel before cleaning up
      mkdir -p dist/fbgemm_gpu
      cp fbgemm/fbgemm_gpu/dist/*.whl dist/fbgemm_gpu
    fi

    for file in "${wheel_dir}"/*.whl
    do
      pip_install_whl "${file}"
    done

    rm -rf fbgemm
  else
    pip_build_and_install "git+https://github.com/pytorch/torchrec.git@${torchrec_commit}" dist/torchrec
    pip_build_and_install "git+https://github.com/pytorch/FBGEMM.git@${fbgemm_commit}#subdirectory=fbgemm_gpu" dist/fbgemm_gpu
  fi
}

function clone_pytorch_xla() {
  if [[ ! -d ./xla ]]; then
    git clone --recursive --quiet https://github.com/pytorch/xla.git
    pushd xla
    # pin the xla hash so that we don't get broken by changes to xla
    git checkout "$(cat ../.github/ci_commit_pins/xla.txt)"
    git submodule sync
    git submodule update --init --recursive
    popd
  fi
}

function install_torchao() {
  local commit
  commit=$(get_pinned_commit torchao)
  pip_build_and_install "git+https://github.com/pytorch/ao.git@${commit}" dist/ao
}

function install_flash_attn_cute() {
  echo "Installing FlashAttention CuTe from GitHub..."
  # Grab latest main til we have a pinned commit
  local flash_attn_commit
  flash_attn_commit=$(git ls-remote https://github.com/Dao-AILab/flash-attention.git HEAD | cut -f1)

  # Clone the repo to a temporary directory
  rm -rf flash-attention-build
  git clone --depth 1 --recursive https://github.com/Dao-AILab/flash-attention.git flash-attention-build

  pushd flash-attention-build
  git checkout "${flash_attn_commit}"

  # Install only the 'cute' sub-directory
  pip_install -e flash_attn/cute/
  popd

  # remove the local repo
  rm -rf flash-attention-build
  echo "FlashAttention CuTe installation complete."
}

function print_sccache_stats() {
  echo 'PyTorch Build Statistics'
  sccache --show-stats

  if [[ -n "${OUR_GITHUB_JOB_ID}" ]]; then
    sccache --show-stats --stats-format json | jq .stats \
      > "sccache-stats-${BUILD_ENVIRONMENT}-${OUR_GITHUB_JOB_ID}.json"
  else
    echo "env var OUR_GITHUB_JOB_ID not set, will not write sccache stats to json"
  fi
}
