#!/bin/bash

. ./common.sh

test_gpu_speed_mnist () {
  echo "Testing: MNIST, GPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/pytorch/examples.git -b perftests

  cd examples/mnist

  conda install -c pytorch torchvision

  # Download data
  python main.py --epochs 0

  SAMPLE_ARRAY=()
  NUM_RUNS=$1

  # Needs warm up to get accurate number
  python main.py --epochs 1 --no-log

  for (( i=1; i<=NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command python main.py --epochs 1 --no-log)
    echo $runtime
    SAMPLE_ARRAY+=("${runtime}")
  done

  cd ../..

  stats=$(python ../get_stats.py "${SAMPLE_ARRAY[@]}")
  echo "Runtime stats in seconds:"
  echo $stats

  if [ "$2" == "compare_with_baseline" ]; then
    python ../compare_with_baseline.py --test-name ${FUNCNAME[0]} --sample-stats "${stats}"
  elif [ "$2" == "compare_and_update" ]; then
    python ../compare_with_baseline.py --test-name ${FUNCNAME[0]} --sample-stats "${stats}" --update
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test test_gpu_speed_mnist "$@"
fi
