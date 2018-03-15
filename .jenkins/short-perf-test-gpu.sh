#!/bin/bash

COMPACT_JOB_NAME="short-perf-test-gpu"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

pip install GitPython sqlalchemy psycopg2-binary

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Running GPU perf test for PyTorch..."

# Get last master commit hash
export PYTORCH_COMMIT_ID=$(git log --format="%H" -n 1)

# Get baseline data from database
if [ -z ${BUILD_ID} ]; then
    python get_baseline.py --local --testtype gpu_runtime --datafile perf_test_numbers_gpu.json
else
    python get_baseline.py --username ${USERNAME} --password ${PASSWORD} --hostname ${DBHOSTNAME} --dbname ${DBNAME} --testtype gpu_runtime --datafile perf_test_numbers_gpu.json
fi

if [[ "$COMMIT_SOURCE" == *master* ]]; then
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
if [[ "$COMMIT_SOURCE" == *master* ]]; then
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

# Push new baseline data to database
if [[ "$COMMIT_SOURCE" == *master* ]]; then
    if [ -z ${BUILD_ID} ]; then
        python update_baseline.py --local --testtype gpu_runtime --datafile new_perf_test_numbers_gpu.json
    else
        python update_baseline.py --username ${USERNAME} --password ${PASSWORD} --hostname ${DBHOSTNAME} --dbname ${DBNAME} --testtype gpu_runtime --datafile new_perf_test_numbers_gpu.json
    fi
fi
