#!/bin/bash

set -ex

export PATH=/opt/conda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Running GPU perf test for PyTorch..."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd test/

# TODO: Move this into a separate file
cat >compare_with_baseline.py << EOL

import sys
import numpy
from scipy import stats

mean_values = {
  "commit": "e2127ef0d24103cc872a64515ce6e54f886941c5",

  "test_gpu_speed_mnist": "13.76155",
  "test_gpu_speed_word_language_model": "115.5332",
  "test_gpu_speed_cudnn_lstm": "4.9698",
  "test_gpu_speed_lstm": "5.15325",
  "test_gpu_speed_mlstm": "4.04270",
}

sigma_values = {
  "commit": "e2127ef0d24103cc872a64515ce6e54f886941c5",

  "test_gpu_speed_mnist": "0.42087",
  "test_gpu_speed_word_language_model": "0.10897",
  "test_gpu_speed_cudnn_lstm": "0.03257",
  "test_gpu_speed_lstm": "0.0725",
  "test_gpu_speed_mlstm": "0.03276",
}

mean = float(mean_values[sys.argv[1]])
sigma = float(sigma_values[sys.argv[1]])

print("baseline mean: ", mean)
print("baseline sigma: ", sigma)

sample_data_list = sys.argv[2:]
sample_data_list = [float(v.strip()) for v in sample_data_list]

sample_mean = numpy.mean(sample_data_list)
sample_sigma = numpy.std(sample_data_list)

print("test mean: ", sample_mean)
print("test sigma: ", sample_sigma)

z_value = (sample_mean - mean) / sigma

print("z-value: ", z_value)

if z_value >= 3:
  raise Exception("z-value >= 3, there is 99.7% chance of perf regression.")
else:
  print("z-value < 3, no perf regression detected.")

EOL

run_test () {
  mkdir test_tmp/ && cd test_tmp/
  $1
  cd .. && rm -rf test_tmp/
}

get_runtime_of_command () {
  TIMEFORMAT=%R

  # runtime=$( { time ($1 &> /dev/null); } 2>&1 1>/dev/null)
  runtime=$( { time $1; } 2>&1 1>/dev/null)
  runtime=${runtime#+++ $1}
  runtime=$(python -c "print($runtime)")

  echo $runtime
}

# Define tests
test_gpu_speed_mnist () {
  echo "Testing: MNIST, GPU"

  git clone https://github.com/yf225/examples.git -b benchmark_test

  cd examples/mnist

  pip install -r requirements.txt

  # Download data
  python main.py --epochs 0

  SAMPLE_ARRAY=()
  NUM_RUNS=5

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python main.py --epochs 1 --no-log")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../..

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

test_gpu_speed_word_language_model () {
  echo "Testing: word language model on Wikitext-2, GPU"

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

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

test_gpu_speed_cudnn_lstm () {
  echo "Testing: CuDNN LSTM, GPU"

  git clone https://github.com/yf225/benchmark.git

  cd benchmark/scripts/

  SAMPLE_ARRAY=()
  NUM_RUNS=5

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python cudnn_lstm.py")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../..

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

test_gpu_speed_lstm () {
  echo "Testing: LSTM, GPU"

  git clone https://github.com/yf225/benchmark.git

  cd benchmark/scripts/

  SAMPLE_ARRAY=()
  NUM_RUNS=5

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python lstm.py")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../..

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

test_gpu_speed_mlstm () {
  echo "Testing: MLSTM, GPU"

  git clone https://github.com/yf225/benchmark.git

  cd benchmark/scripts/

  SAMPLE_ARRAY=()
  NUM_RUNS=5

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python mlstm.py")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  cd ../..

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

echo "ENTERED_USER_LAND"

# Run tests
run_test test_gpu_speed_mnist
run_test test_gpu_speed_word_language_model
run_test test_gpu_speed_cudnn_lstm
run_test test_gpu_speed_lstm
run_test test_gpu_speed_mlstm

echo "EXITED_USER_LAND"
