#!/bin/bash

run_test () {
  rm -rf test_tmp/ && mkdir test_tmp/ && cd test_tmp/
  "$@"
  cd .. && rm -rf test_tmp/
}

get_runtime_of_command () {
  TIMEFORMAT=%R

  # runtime=$( { time ($@ &> /dev/null); } 2>&1 1>/dev/null)
  runtime=$( { time "$@"; } 2>&1 1>/dev/null)
  if [[ $runtime == *"Error"* ]]; then
    exit 1
  fi
  runtime=${runtime#+++ $@}
  runtime=$(python -c "print($runtime)")

  echo $runtime
}
