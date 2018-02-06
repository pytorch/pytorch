. ./common.sh

test_cpu_speed_mini_sequence_labeler () {
  echo "Testing: mini sequence labeler, CPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  curl https://gist.githubusercontent.com/yf225/40c0adb8bfb2a7b774fa266fb4bc409e/raw/20c67ebadbd75f41c6c9fd2e00b4b2562b60700a/mini_sequence_labeler.py -O
  curl https://gist.githubusercontent.com/yf225/592b39ca6a3fc835a4d1532fb1474d26/raw/76f57c198cb7afdc5122e413c2a3023ed024b643/wsj.pkl -O

  SAMPLE_ARRAY=()
  NUM_RUNS=20

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python mini_sequence_labeler.py")
    SAMPLE_ARRAY+=(${runtime})
  done

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

