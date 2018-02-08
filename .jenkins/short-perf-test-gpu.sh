#!/bin/bash

set -ex

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Running GPU perf test for PyTorch..."

echo "ENTERED_USER_LAND"

# Include tests
. ./test_gpu_speed_mnist.sh
. ./test_gpu_speed_word_language_model.sh
. ./test_gpu_speed_cudnn_lstm.sh
. ./test_gpu_speed_lstm.sh
. ./test_gpu_speed_mlstm.sh

# Run tests
run_test test_gpu_speed_mnist compare_with_baseline
run_test test_gpu_speed_word_language_model compare_with_baseline
run_test test_gpu_speed_cudnn_lstm compare_with_baseline
run_test test_gpu_speed_lstm compare_with_baseline
run_test test_gpu_speed_mlstm compare_with_baseline

echo "EXITED_USER_LAND"
