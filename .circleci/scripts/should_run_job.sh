#!/usr/bin/env bash
set -exu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Check if we should actually run
echo "BUILD_ENVIRONMENT: ${BUILD_ENVIRONMENT:-}"
echo "CIRCLE_PULL_REQUEST: ${CIRCLE_PULL_REQUEST:-}"
if [ -z "${BUILD_ENVIRONMENT:-}" ]; then
  echo "Cannot run should_run_job.sh if BUILD_ENVIRONMENT is not defined!"
  echo "CircleCI scripts are probably misconfigured."
  exit 1
fi
if ! [ -e "$SCRIPT_DIR/COMMIT_MSG" ]; then
  echo "Cannot run should_run_job.sh if you don't have COMMIT_MSG"
  echo "written out.  Are you perhaps running the wrong copy of this script?"
  echo "You should be running the copy in ~/workspace; SCRIPT_DIR=$SCRIPT_DIR"
  exit 1
fi
if [ -n "${CIRCLE_PULL_REQUEST:-}" ]; then
  if [[ $CIRCLE_BRANCH != "ci-all/"* ]]; then
    # Don't swallow "script doesn't exist
    [ -e "$SCRIPT_DIR/should_run_job.py"  ]
    if ! python "$SCRIPT_DIR/should_run_job.py" "${BUILD_ENVIRONMENT:-}" < "$SCRIPT_DIR/COMMIT_MSG" ; then
      circleci step halt
      exit
    fi
  fi
fi
