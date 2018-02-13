. ./common.sh

test_gpu_speed_word_language_model () {
  echo "Testing: word language model on Wikitext-2, GPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/yf225/examples.git -b benchmark_test

  cd examples/word_language_model

  SAMPLE_ARRAY=()
  NUM_RUNS=5

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python main.py --cuda --epochs 1")
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
  run_test test_gpu_speed_word_language_model "$@"
fi