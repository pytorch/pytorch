. ./common.sh

test_cpu_speed_mnist () {
  echo "Testing: MNIST, CPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/yf225/examples.git -b benchmark_test

  cd examples/mnist

  pip install -r requirements.txt

  # Download data
  python main.py --epochs 0

  SAMPLE_ARRAY=()
  NUM_RUNS=20

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python main.py --epochs 1 --no-log")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../..

  stats=$(python ../get_stats.py ${SAMPLE_ARRAY[@]})
  echo "Runtime stats in seconds:"
  echo $stats

  if [ "$1" == "compare_with_baseline" ]; then
    python ../compare_with_baseline.py ${FUNCNAME[0]} "${stats}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test test_cpu_speed_mnist "$@"
fi

