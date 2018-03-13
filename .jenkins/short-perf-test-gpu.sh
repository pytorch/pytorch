#!/bin/bash

COMPACT_JOB_NAME="short-perf-test-gpu"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Running GPU perf test for PyTorch..."

# Get last master commit hash
export PYTORCH_COMMIT_ID=$(git log --format="%H" -n 1)

# Get baseline file from https://github.com/yf225/perf-tests
if [ -f /var/lib/jenkins/host-workspace/perf_test_numbers_gpu.json ]; then
    cp /var/lib/jenkins/host-workspace/perf_test_numbers_gpu.json perf_test_numbers_gpu.json
else
    curl https://raw.githubusercontent.com/yf225/perf-tests/master/perf_test_numbers_gpu.json -O
fi

if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    # Prepare new baseline file
    cp perf_test_numbers_gpu.json new_perf_test_numbers_gpu.json
    python update_commit_hash.py new_perf_test_numbers_gpu.json ${PYTORCH_COMMIT_ID}
fi

# Include tests
. ./test_gpu_speed_mnist.sh
. ./test_gpu_speed_word_language_model.sh
. ./test_gpu_speed_cudnn_lstm.sh
. ./test_gpu_speed_lstm.sh
. ./test_gpu_speed_mlstm.sh

# Run tests
if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    run_test test_gpu_speed_mnist 20 compare_and_update
    run_test test_gpu_speed_word_language_model 20 compare_and_update
    run_test test_gpu_speed_cudnn_lstm 20 compare_and_update
    run_test test_gpu_speed_lstm 20 compare_and_update
    run_test test_gpu_speed_mlstm 20 compare_and_update
else
    run_test test_gpu_speed_mnist 20 compare_with_baseline
    run_test test_gpu_speed_word_language_model 20 compare_with_baseline
    run_test test_gpu_speed_cudnn_lstm 20 compare_with_baseline
    run_test test_gpu_speed_lstm 20 compare_with_baseline
    run_test test_gpu_speed_mlstm 20 compare_with_baseline
fi

if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    # Push new baseline file
    cp new_perf_test_numbers_gpu.json /var/lib/jenkins/host-workspace/perf_test_numbers_gpu.json
    cd /var/lib/jenkins/host-workspace
    git config --global user.email jenkins@ci.pytorch.org
    git config --global user.name Jenkins
    git add perf_test_numbers_gpu.json
    git commit -m "New GPU perf test baseline from ${PYTORCH_COMMIT_ID}"
fi
