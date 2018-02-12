. ./common.sh

test_cpu_speed_mini_sequence_labeler () {
  echo "Testing: mini sequence labeler, CPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/pytorch/benchmark.git

  cd benchmark/

  git fetch origin pull/14/head:mini_sequence_labeler
  git checkout mini_sequence_labeler

  cd scripts/mini_sequence_labeler

  SAMPLE_ARRAY=()
  NUM_RUNS=20

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python main.py")
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../../..

  stats=$(python ../get_stats.py ${SAMPLE_ARRAY[@]})
  echo "Runtime stats in seconds:"
  echo $stats

  if [ "$1" == "compare_with_baseline" ]; then
    python ../compare_with_baseline.py ${FUNCNAME[0]} "${stats}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test test_cpu_speed_mini_sequence_labeler "$@"
fi

