#!/bin/bash

# Common setup for all Jenkins scripts
# shellcheck source=./common_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"
set -ex

# Save the SCRIPT_DIR absolute path in case later we chdir (as occurs in the gpu perf test)
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )"

# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

# Figure out which Python to use for ROCm
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]] && [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
  unset HIP_PLATFORM
  PYTHON=$(which "python${BASH_REMATCH[1]}")
  # non-interactive bashs do not expand aliases by default
  shopt -s expand_aliases
  export PYTORCH_TEST_WITH_ROCM=1
  alias python='$PYTHON'
  # temporary to locate some kernel issues on the CI nodes
  export HSAKMT_DEBUG_LEVEL=4
fi

# This token is used by a parser on Jenkins logs for determining
# if a failure is a legitimate problem, or a problem with the build
# system; to find out more, grep for this string in ossci-job-dsl.
echo "ENTERED_USER_LAND"

export IS_PYTORCH_CI=1

# compositional trap taken from https://stackoverflow.com/a/7287873/23845

# note: printf is used instead of echo to avoid backslash
# processing and to properly handle values that begin with a '-'.

log() { printf '%s\n' "$*"; }
error() { log "ERROR: $*" >&2; }
fatal() { error "$@"; exit 1; }

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

trap_add cleanup EXIT

if [[ "$BUILD_ENVIRONMENT" != *pytorch-win-* ]]; then
  if which sccache > /dev/null; then
    # Save sccache logs to file
    sccache --stop-server > /dev/null  2>&1 || true
    rm ~/sccache_error.log || true
    if [[ -n "${SKIP_SCCACHE_INITIALIZATION:-}" ]]; then
      # sccache --start-server seems to hang forever on self hosted runners for GHA
      # so let's just go ahead and skip the --start-server altogether since it seems
      # as though sccache still gets used even when the sscache server isn't started
      # explicitly
      echo "Skipping sccache server initialization, setting environment variables"
      export SCCACHE_IDLE_TIMEOUT=1200
      export SCCACHE_ERROR_LOG=~/sccache_error.log
      export RUST_LOG=sccache::server=error
    elif [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
      SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server
    else
      # increasing SCCACHE_IDLE_TIMEOUT so that extension_backend_test.cpp can build after this PR:
      # https://github.com/pytorch/pytorch/pull/16645
      SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=1200 RUST_LOG=sccache::server=error sccache --start-server
    fi

    # Report sccache stats for easier debugging
    sccache --zero-stats
    function sccache_epilogue() {
      echo '=================== sccache compilation log ==================='
      python "$SCRIPT_DIR/print_sccache_log.py" ~/sccache_error.log 2>/dev/null
      echo '=========== If your build fails, please take a look at the log above for possible reasons ==========='
      sccache --show-stats
      sccache --stop-server || true
    }
    trap_add sccache_epilogue EXIT
  fi

  if which ccache > /dev/null; then
    # Report ccache stats for easier debugging
    ccache --zero-stats
    ccache --show-stats
    function ccache_epilogue() {
      ccache --show-stats
    }
    trap_add ccache_epilogue EXIT
  fi
fi

# It's called a COMPACT_JOB_NAME because it's distinct from the
# Jenkin's provided JOB_NAME, which also includes a prefix folder
# e.g. pytorch-builds/

if [ -z "$COMPACT_JOB_NAME" ]; then
  echo "Jenkins build scripts must set COMPACT_JOB_NAME"
  exit 1
fi

if [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda10.1-cudnn7-py3* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-trusty-py3.6-gcc7* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch_macos* ]]; then
  BUILD_TEST_LIBTORCH=1
else
  # shellcheck disable=SC2034
  BUILD_TEST_LIBTORCH=0
fi

# Use conda cmake in some CI build. Conda cmake will be newer than our supported
# min version (3.5 for xenial and 3.10 for bionic),
# so we only do it in four builds that we know should use conda.
# Linux bionic cannot find conda mkl with cmake 3.10, so we need a cmake from conda.
# Alternatively we could point cmake to the right place
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
if [[ "$BUILD_ENVIRONMENT" == *pytorch-xla-linux-bionic* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda9-cudnn7-py2* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda10.1-cudnn7-py3* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-*centos* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-bionic* ]]; then
  if ! which conda; then
    echo "Expected ${BUILD_ENVIRONMENT} to use conda, but 'which conda' returns empty"
    exit 1
  else
    conda install -q -y cmake
  fi
  if [[ "$BUILD_ENVIRONMENT" == *pytorch-*centos* ]]; then
    # cmake3 package will conflict with conda cmake
    sudo yum -y remove cmake3 || true
  fi
fi

retry () {
  "$@"  || (sleep 1 && "$@") || (sleep 2 && "$@")
}
