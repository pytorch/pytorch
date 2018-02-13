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

trap cleanup EXIT

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

if grep --line-regexp -q "$COMPACT_JOB_NAME" "$(dirname "${BASH_SOURCE[0]}")/enabled-configs.txt"; then
  echo "Test is enabled, proceeding"
else
  echo "Test is disabled, FAILING now (revert changes to enabled-configs.txt to fix this)"
  exit 1
fi
