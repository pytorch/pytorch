#!/bin/bash

COMPACT_JOB_NAME="short-perf-test-gpu"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

echo "Running GPU perf test for PyTorch..."

pip install awscli

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

if [[ "$COMMIT_SOURCE" == *master* ]]; then
    # Get current master commit hash
    export MASTER_COMMIT_ID=$(git log --format="%H" -n 1)
fi

# Get baseline file from ossci-perf-test S3 bucket
aws s3 cp s3://ossci-perf-test/pytorch/gpu_runtime/LATEST_TESTED_COMMIT LATEST_TESTED_COMMIT
export LATEST_TESTED_COMMIT="$(cat LATEST_TESTED_COMMIT)"
aws s3 cp s3://ossci-perf-test/pytorch/gpu_runtime/${LATEST_TESTED_COMMIT}.json gpu_runtime.json

if [[ "$COMMIT_SOURCE" == *master* ]]; then
    # Prepare new baseline file
    cp gpu_runtime.json new_gpu_runtime.json
    python update_commit_hash.py new_gpu_runtime.json ${MASTER_COMMIT_ID}
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

if [[ "$COMMIT_SOURCE" == *master* ]]; then
    aws s3 cp new_gpu_runtime.json s3://ossci-perf-test/pytorch/gpu_runtime/${MASTER_COMMIT_ID}.json --acl public-read

    # If current commit is newer, then we put it in LATEST_TESTED_COMMIT
    aws s3 cp s3://ossci-perf-test/pytorch/gpu_runtime/LATEST_TESTED_COMMIT LATEST_TESTED_COMMIT
    export LATEST_TESTED_COMMIT="$(cat LATEST_TESTED_COMMIT)"
    if ! git merge-base --is-ancestor ${MASTER_COMMIT_ID} ${LATEST_TESTED_COMMIT}; then
        echo "${MASTER_COMMIT_ID}" > LATEST_TESTED_COMMIT
        aws s3 cp LATEST_TESTED_COMMIT s3://ossci-perf-test/pytorch/gpu_runtime/LATEST_TESTED_COMMIT --acl public-read
    fi
fi
