#!/bin/bash

. ./common.sh

test_cpu_speed_torch () {
  echo "Testing: torch.*, CPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/yf225/perf-tests.git

  if [ "$1" == "compare_with_baseline" ]; then
    export ARGS="--compare ../cpu_runtime.json"
  elif [ "$1" == "compare_and_update" ]; then
    export ARGS="--compare ../cpu_runtime.json --update ../new_cpu_runtime.json"
  elif [ "$1" == "update_only" ]; then
    export ARGS="--update ../new_cpu_runtime.json"
  fi

  if ! python perf-tests/modules/test_cpu_torch.py ${ARGS}; then
    echo "To reproduce this regression, run \`cd .jenkins/pytorch/perf_test/ && bash ${FUNCNAME[0]}.sh\` on your local machine and compare the runtime before/after your code change."
    exit 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test test_cpu_speed_torch "$@"
fi

