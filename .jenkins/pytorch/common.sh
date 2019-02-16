#!/bin/bash

# Common setup for all Jenkins scripts

# NB: define this function before set -x, so that we don't
# pollute the log with a premature EXITED_USER_LAND ;)
function cleanup {
  # Note that if you've exited user land, then CI will conclude that
  # any failure is the CI's fault.  So we MUST only output this
  # string
  retcode=$?
  set +x
  if [ $retcode -eq 0 ]; then
    echo "EXITED_USER_LAND"
  fi
}

set -ex

# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

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
    trap_add_cmd=$1; shift || fatal "${FUNCNAME} usage error"
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

function assert_git_not_dirty() {
    # TODO: we should add an option to `build_amd.py` that reverts the repo to
    #       an unmodified state.
    if ([[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *xla* ]]) ; then
        git_status=$(git status --porcelain)
        if [[ $git_status ]]; then
            echo "Build left local git repository checkout dirty"
            echo "git status --porcelain:"
            echo "${git_status}"
            exit 1
        fi
    fi
}

if which sccache > /dev/null; then
  # Save sccache logs to file
  sccache --stop-server || true
  rm ~/sccache_error.log || true
  # increasing SCCACHE_IDLE_TIMEOUT so that extension_backend_test.cpp can build after this PR:
  # https://github.com/pytorch/pytorch/pull/16645
  SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=1200 RUST_LOG=sccache::server=error sccache --start-server

  # Report sccache stats for easier debugging
  sccache --zero-stats
  function sccache_epilogue() {
    echo '=================== sccache compilation log ==================='
    python $(dirname "${BASH_SOURCE[0]}")/print_sccache_log.py ~/sccache_error.log
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

# It's called a COMPACT_JOB_NAME because it's distinct from the
# Jenkin's provided JOB_NAME, which also includes a prefix folder
# e.g. pytorch-builds/

if [ -z "$COMPACT_JOB_NAME" ]; then
  echo "Jenkins build scripts must set COMPACT_JOB_NAME"
  exit 1
fi

if grep --line-regexp -q "$COMPACT_JOB_NAME" "$(dirname "${BASH_SOURCE[0]}")/disabled-configs.txt"; then
  echo "Job is explicitly disabled, SKIPPING"
  exit 0
else
  echo "Job is not disabled, proceeding"
fi

if grep --line-regexp -q "$COMPACT_JOB_NAME" "$(dirname "${BASH_SOURCE[0]}")/enabled-configs.txt"; then
  echo "Job is enabled, proceeding"
else
  echo "Job is not enabled, FAILING now (revert changes to enabled-configs.txt to fix this)"
  exit 1
fi

if [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda9-cudnn7-py3 ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-trusty-py3.6-gcc7* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch_macos* ]]; then
  BUILD_TEST_LIBTORCH=1
else
  BUILD_TEST_LIBTORCH=0
fi

# Use conda cmake in some CI build. Conda cmake will be newer than our supported
# min version 3.5, so we only do it in two builds that we know should use conda.
if [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda* ]]; then
  if [[ "$BUILD_ENVIRONMENT" == *cuda8-cudnn7-py2* ]] || \
     [[ "$BUILD_ENVIRONMENT" == *cuda9-cudnn7-py3* ]]; then
    if ! which conda; then
      echo "Expected ${BUILD_ENVIRONMENT} to use conda, but 'which conda' returns empty"
      exit 1
    else
      conda install -q -y cmake
    fi
  else
    if ! cmake --version | grep 'cmake version 3\.5'; then
      echo "Expected ${BUILD_ENVIRONMENT} to have cmake version 3.5.* (min support version), but 'cmake --version' returns:"
      cmake --version
      exit 1
    fi
  fi
fi

function get_exit_code() {
  set +e
  "$@"
  retcode=$?
  set -e
  return $retcode
}
