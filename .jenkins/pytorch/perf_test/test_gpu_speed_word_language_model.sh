#!/bin/bash

. ./common.sh

test_gpu_speed_word_language_model () {
  echo "Testing: word language model on Wikitext-2, GPU"

  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  git clone https://github.com/pytorch/examples.git -b perftests

  cd examples/word_language_model

  cd data/wikitext-2

  # Reduce dataset size, so that we can have more runs per test
  sed -n '1,200p' test.txt > test_tmp.txt
  sed -n '1,1000p' train.txt > train_tmp.txt
  sed -n '1,200p' valid.txt > valid_tmp.txt

  mv test_tmp.txt test.txt
  mv train_tmp.txt train.txt
  mv valid_tmp.txt valid.txt

  cd ../..

  SAMPLE_ARRAY=()
  NUM_RUNS=$1

  for (( i=1; i<=NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command python main.py --cuda --epochs 1)
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
  run_test test_gpu_speed_word_language_model "$@"
fi
