#!/bin/bash
set -e

. ./common.sh

test_gpu_speed_cudnn_lstm () {
  echo "Testing: CuDNN LSTM, GPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/pytorch/benchmark.git

  cd benchmark/

  git checkout 43dfb2c0370e70ef37f249dc09aff9f0ccd2ddb0

  cd scripts/

  SAMPLE_ARRAY=()
  NUM_RUNS=$1

  for (( i=1; i<=NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command python cudnn_lstm.py --skip-cpu-governor-check)
    echo "$runtime"
    SAMPLE_ARRAY+=("${runtime}")
  done

  cd ../..

  stats=$(python ../get_stats.py "${SAMPLE_ARRAY[@]}")
  echo "Runtime stats in seconds:"
  echo "$stats"

  if [ "$2" == "compare_with_baseline" ]; then
    python ../compare_with_baseline.py --test-name "${FUNCNAME[0]}" --sample-stats "${stats}"
  elif [ "$2" == "compare_and_update" ]; then
    python ../compare_with_baseline.py --test-name "${FUNCNAME[0]}" --sample-stats "${stats}" --update
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test test_gpu_speed_cudnn_lstm "$@"
fi
