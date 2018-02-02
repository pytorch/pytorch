#!/bin/bash

set -ex

export PATH=/opt/conda/bin:$PATH

echo "Running CPU perf test for PyTorch..."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd test/

# TODO: Move this into its own file
cat >compare_with_baseline.py << EOL

import sys
import numpy
from scipy import stats

mean_values = {
  "commit": "92aeca1279265d24493dc6ced7dde9a368faf048",

  "test_cpu_speed_mini_sequence_labeler": "2.62557",
  "test_cpu_speed_mnist": "18.79468",
}

sigma_values = {
  "commit": "92aeca1279265d24493dc6ced7dde9a368faf048",

  "test_cpu_speed_mini_sequence_labeler": "0.12167",
  "test_cpu_speed_mnist": "2.37561",
}

mean = float(mean_values[sys.argv[1]])
sigma = float(sigma_values[sys.argv[1]])

print("population mean: ", mean)
print("population sigma: ", sigma)

sample_data_list = sys.argv[2:]
sample_data_list = [float(v.strip()) for v in sample_data_list]

sample_mean = numpy.mean(sample_data_list)

print("sample mean: ", sample_mean)

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
test_cpu_speed_mini_sequence_labeler () {
  echo "Testing: mini sequence labeler, CPU"

  curl https://gist.githubusercontent.com/yf225/40c0adb8bfb2a7b774fa266fb4bc409e/raw/20c67ebadbd75f41c6c9fd2e00b4b2562b60700a/mini_sequence_labeler.py -O
  curl https://gist.githubusercontent.com/yf225/592b39ca6a3fc835a4d1532fb1474d26/raw/76f57c198cb7afdc5122e413c2a3023ed024b643/wsj.pkl -O

  SAMPLE_ARRAY=()
  NUM_RUNS=20

  for (( i=1; i<=$NUM_RUNS; i++ )) do
    runtime=$(get_runtime_of_command "python mini_sequence_labeler.py")
    echo $runtime
    SAMPLE_ARRAY+=(${runtime})
  done

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

test_cpu_speed_mnist () {
  echo "Testing: MNIST, CPU"

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

  python ../compare_with_baseline.py ${FUNCNAME[0]} "${SAMPLE_ARRAY[@]}"
}

echo "ENTERED_USER_LAND"

# Run tests
run_test test_cpu_speed_mini_sequence_labeler
run_test test_cpu_speed_mnist

echo "EXITED_USER_LAND"
