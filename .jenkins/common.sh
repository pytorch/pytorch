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
#   $JOB_NAME

# This token is used by a parser on Jenkins logs for determining
# if a failure is a legitimate problem, or a problem with the build
# system; to find out more, grep for this string in ossci-job-dsl.
echo "ENTERED_USER_LAND"

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

# Converts:
# pytorch-builds/pytorch-macos-10.13-py3-build-test ==> pytorch-macos-10.13-py3-build-test
#
# We don't match the full path so that you can make a job in a subfolder
# that will trigger this checking.
#
# NB: This greedily matches until the last /, so if you ever decide to
# restructure the PyTorch jobs to have some more directory hierarchy,
# you will have to adjust this.
COMPACT_JOB_NAME="$(echo "$JOB_NAME" | perl -pe 's{^(?:.+/)?(.+)$}{$1}o')"

if grep --line-regexp -q "$COMPACT_JOB_NAME" "$(dirname "${BASH_SOURCE[0]}")/disabled-configs.txt"; then
  echo "Test is explicitly disabled, SKIPPING"
  exit 0
else
  echo "Test is not disabled, proceeding"
fi

if grep --line-regexp -q "$COMPACT_JOB_NAME" "$(dirname "${BASH_SOURCE[0]}")/enabled-configs.txt"; then
  echo "Test is enabled, proceeding"
else
  echo "Test not enabled, FAILING now (revert changes to enabled-configs.txt to fix this)"
  exit 1
fi
